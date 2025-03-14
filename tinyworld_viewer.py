import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import random

class PanoramicViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Panoramic/Spherical Image Viewer")
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Image state and view parameters.
        self.image = None             # Original PIL image
        self.img_width = 0            # Image width (set dynamically when opened)
        self.img_height = 0           # Image height (set dynamically when opened)
        self.zoom = 1.0               # Current zoom factor (1.0 means no zoom)
        self.full_view_mode = False   # If True, display full image scaled to canvas.
        self.offset_x = 0             # Horizontal offset for panning (wraps around)
        self.offset_y = 0             # Vertical offset for panning (clamped)
        self.start_x = None           # Starting x coordinate for dragging
        self.start_y = None           # Starting y coordinate for dragging

        # Auto-motion configuration (sweeping pan, tilt, and zoom).
        self.auto_motion_active = True  # Auto-motion remains active until user interaction.
        self.auto_motion_interval = 8    # Update interval in milliseconds (20ms = 50fps).
        self.auto_duration = 750          # Number of updates per transition.
        self.auto_progress = 0.0          # Progress from 0 to 1 for the current transition.

        # Starting state for the current sweep transition.
        self.auto_start_offset_x = 0.0
        self.auto_start_offset_y = 0.0
        self.auto_start_zoom = 1.0

        # Target state for the sweep transition.
        self.auto_target_offset_x = 0.0
        self.auto_target_offset_y = 0.0
        self.auto_target_zoom = 1.0

        # Allowed zoom limits for auto-motion.
        self.auto_zoom_min = 0.5      # Minimum zoom factor.
        self.auto_zoom_max = 2.0      # Maximum zoom factor.

        # Create a menu bar with File and View menus.
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        view_menu.add_command(label="Distorted Full View", command=self.full_view)
        menubar.add_cascade(label="View", menu=view_menu)
        self.config(menu=menubar)

        # Bind mouse events for dragging (panning).
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Bind mouse scroll events for zooming (Windows and Linux).
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # For Linux scroll up.
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # For Linux scroll down.

        # Bind keyboard shortcuts for zooming (Ctrl + and Ctrl -).
        self.bind_all("<Control-Key-plus>", lambda e: self.zoom_in())
        self.bind_all("<Control-Key-minus>", lambda e: self.zoom_out())
        # Disable auto-motion on any key press.
        self.bind_all("<Key>", lambda e: self.disable_auto_motion())

        # Start the auto-motion loop.
        self.after(self.auto_motion_interval, self.auto_motion)

    def disable_auto_motion(self):
        """Disable automatic sweeping motions when the user interacts."""
        if self.auto_motion_active:
            self.auto_motion_active = False

    def open_image(self):
        """Open an image file, store its dimensions dynamically, and reset view parameters."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.image = Image.open(file_path)
            self.img_width, self.img_height = self.image.size  # Store dynamic dimensions.
            self.offset_x = 0
            self.offset_y = 0
            self.zoom = 1.0
            self.full_view_mode = False
            # Restart auto-motion until user interaction.
            self.auto_motion_active = True
            self.auto_progress = 0.0
            self.redraw()

    def choose_new_target(self):
        """Choose a new random target for pan, tilt, and zoom over the whole image."""
        if not self.image:
            return
        # Store the current state as the starting state.
        self.auto_start_offset_x = self.offset_x
        self.auto_start_offset_y = self.offset_y
        self.auto_start_zoom = self.zoom

        # Choose a new random zoom factor.
        self.auto_target_zoom = random.uniform(self.auto_zoom_min, self.auto_zoom_max)

        # For horizontal offset, allow any value (wrap-around is supported).
        self.auto_target_offset_x = random.uniform(0, self.img_width)

        # For vertical offset, ensure the cropped area remains within the image.
        crop_h = self.canvas_height / self.auto_target_zoom
        max_offset_y = max(0, self.img_height - crop_h)
        self.auto_target_offset_y = random.uniform(0, max_offset_y) if max_offset_y > 0 else 0

    def ease(self, t):
        """Cosine-based ease in/out function."""
        return 0.5 - 0.5 * math.cos(math.pi * t)

    def redraw(self):
        """Redraw the image on the canvas based on current offsets, zoom, and view mode."""
        if not self.image:
            return

        # Use stored dynamic image dimensions.
        img_width, img_height = self.img_width, self.img_height

        if self.full_view_mode:
            # Scale the full image to the canvas (may distort aspect ratio).
            full_image = self.image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(full_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            return

        # Compute the cropping region based on the current zoom.
        crop_w = self.canvas_width / self.zoom
        crop_h = self.canvas_height / self.zoom
        int_crop_w = int(crop_w)
        int_crop_h = int(crop_h)

        # Horizontal panning uses wrap-around.
        mod_offset_x = int(self.offset_x % img_width)
        # Vertical offset is clamped.
        max_offset_y = max(0, img_height - int_crop_h)
        clamped_offset_y = int(max(0, min(self.offset_y, max_offset_y)))

        # Create a new image to assemble the current view.
        view = Image.new("RGB", (int_crop_w, int_crop_h))

        if mod_offset_x + int_crop_w <= img_width:
            box = (mod_offset_x, clamped_offset_y, mod_offset_x + int_crop_w, clamped_offset_y + int_crop_h)
            piece = self.image.crop(box)
            view.paste(piece, (0, 0))
        else:
            # When the crop region exceeds the image width, combine two crops.
            width1 = img_width - mod_offset_x
            box1 = (mod_offset_x, clamped_offset_y, img_width, clamped_offset_y + int_crop_h)
            piece1 = self.image.crop(box1)
            view.paste(piece1, (0, 0))

            width2 = int_crop_w - width1
            box2 = (0, clamped_offset_y, width2, clamped_offset_y + int_crop_h)
            piece2 = self.image.crop(box2)
            view.paste(piece2, (width1, 0))

        resized_view = view.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_view)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def zoom_in(self):
        """Zoom in and disable auto-motion."""
        self.disable_auto_motion()
        self.full_view_mode = False
        self.zoom *= 1.25
        self.redraw()

    def zoom_out(self):
        """Zoom out and disable auto-motion."""
        self.disable_auto_motion()
        self.full_view_mode = False
        self.zoom /= 1.25
        self.redraw()

    def reset_zoom(self):
        """Reset zoom to 1.0 and disable auto-motion."""
        self.disable_auto_motion()
        self.full_view_mode = False
        self.zoom = 1.0
        self.redraw()

    def full_view(self):
        """Switch to full view mode (display full image scaled to canvas)."""
        self.disable_auto_motion()
        self.full_view_mode = True
        self.redraw()

    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming."""
        self.disable_auto_motion()
        if hasattr(event, 'delta'):
            if event.delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        elif event.num == 4:
            self.zoom_in()
        elif event.num == 5:
            self.zoom_out()

    def on_button_press(self, event):
        """Record the mouse position when pressing (disable auto-motion)."""
        self.disable_auto_motion()
        if self.full_view_mode:
            return
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        """Update offsets based on mouse drag (disable auto-motion)."""
        self.disable_auto_motion()
        if self.full_view_mode:
            return
        if self.start_x is not None and self.start_y is not None:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
            self.offset_x -= dx / self.zoom
            self.offset_y -= dy / self.zoom
            self.start_x = event.x
            self.start_y = event.y
            self.redraw()

    def on_button_release(self, event):
        """Reset mouse drag starting positions."""
        self.start_x = None
        self.start_y = None

    def auto_motion(self):
        """Perform automatic sweeping pan, tilt, and zoom if no user interaction occurs."""
        if self.auto_motion_active and self.image and not self.full_view_mode:
            # If starting a new transition, choose a new target.
            if self.auto_progress <= 0.0:
                self.choose_new_target()

            # Increment progress.
            self.auto_progress += 1.0 / self.auto_duration
            if self.auto_progress > 1.0:
                self.auto_progress = 1.0

            # Compute eased progress.
            t_ease = self.ease(self.auto_progress)

            # Interpolate horizontal offset (with wrap-around).
            delta_x = self.auto_target_offset_x - self.auto_start_offset_x
            if abs(delta_x) > self.img_width / 2:
                if delta_x > 0:
                    delta_x -= self.img_width
                else:
                    delta_x += self.img_width
            self.offset_x = self.auto_start_offset_x + delta_x * t_ease

            # Interpolate vertical offset.
            self.offset_y = self.auto_start_offset_y + (self.auto_target_offset_y - self.auto_start_offset_y) * t_ease

            # Interpolate zoom.
            self.zoom = self.auto_start_zoom + (self.auto_target_zoom - self.auto_start_zoom) * t_ease

            self.redraw()

            # If transition is complete, reset progress to start a new one.
            if self.auto_progress >= 1.0:
                self.auto_progress = 0.0

        self.after(self.auto_motion_interval, self.auto_motion)

if __name__ == "__main__":
    app = PanoramicViewer()
    app.mainloop()
