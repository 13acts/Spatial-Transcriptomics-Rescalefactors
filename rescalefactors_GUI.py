import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import anndata

from shapely.geometry import MultiPoint, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from shapely.geometry import mapping
# import matplotlib.pyplot as plt


class CollapsibleSection(tk.Frame):
    def __init__(self, parent, title="Section", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.is_expanded = False

        self.toggle_btn = tk.Button(self, text="▶ " + title, command=self.toggle)
        self.toggle_btn.pack(anchor="w")

        self.content = tk.Frame(self)
        self.content.pack(fill="x", expand=True)
        self.content.forget()  # hide initially

    def toggle(self):
        if self.is_expanded:
            self.content.forget()
            self.toggle_btn.config(text="▶ " + self.toggle_btn.cget("text")[2:])
        else:
            self.content.pack(fill="x", expand=True)
            self.toggle_btn.config(text="▼ " + self.toggle_btn.cget("text")[2:])
        self.is_expanded = not self.is_expanded


class SpotOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visium Spot Overlay Viewer")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.max_display_size = (screen_height*0.9, screen_width*0.9)

        # Use grid layout
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)

        # Image frame (left)
        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=0, sticky="nsew")

        # Canvas inside image frame
        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Drag n drop
        self._dragging = False
        self._drag_start = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Control panel (right)
        # Container for the scrollable side panel
        side_container = tk.Frame(root, width=250)
        side_container.grid(row=0, column=1, sticky="ns")
        side_container.grid_propagate(False)  # Prevent it from resizing to contents

        # Canvas and scrollbar inside side container
        side_canvas = tk.Canvas(side_container, width=250, borderwidth=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(side_container, orient="vertical", command=side_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        side_canvas.pack(side="left", fill="both", expand=True)

        # Frame that holds the actual controls
        self.control_frame = tk.Frame(side_canvas, padx=10, pady=10)
        side_window = side_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        def on_canvas_configure(event):
            # Match embedded frame width to canvas width
            side_canvas.itemconfig(side_window, width=event.width)
        side_canvas.bind("<Configure>", on_canvas_configure)

        # Configure scrolling region
        def on_frame_configure(event):
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
        self.control_frame.bind("<Configure>", on_frame_configure)
        side_canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            side_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        side_canvas.bind_all("<MouseWheel>", _on_mousewheel)


        # Load Data button
        load_btn = tk.Button(self.control_frame, text="Load AnnData", command=self.load_anndata)
        load_btn.pack(pady=10, fill="x")

        # Transformation sliders
        tk.Button(self.control_frame, text="Reset to Default", command=self.reset_transformations).pack(pady=10, fill="x")

        self.show_spots = tk.IntVar(value=1)
        tk.Checkbutton(
            self.control_frame,
            text="Show Spots",
            variable=self.show_spots,
            command=self.redraw
        ).pack(anchor="w")

        # self.show_clusters = tk.IntVar()
        # tk.Checkbutton(
        #     self.control_frame,
        #     text="Show Cluster Outlines",
        #     variable=self.show_clusters,
        #     command=self.redraw
        # ).pack(anchor="w")
        # self.shapely_alpha = self.create_slider_with_entry("alpha", 0.01, 10, 0.1, resolution=0.01)

        self.scalef_multiplier_log2 = self.create_momentum_slider("scalef multiplier", initial=0.0, speed=0.2)
        self.shift_x = self.create_momentum_slider("Shift X", initial=0.0, speed=100)
        self.shift_y = self.create_momentum_slider("Shift Y", initial=0.0, speed=100)

        # Advanced
        transform_section = CollapsibleSection(self.control_frame, title="Advanced")
        transform_section.pack(fill="x", pady=5)

        # Add transform widgets into section.content
        self.spot_radius_multiplier_log2 = self.create_momentum_slider("spot_diameter_fullres multiplier", initial=0.0, speed=0.2, parent=transform_section.content)
        self.scale_x_log2 = self.create_momentum_slider("Scale X", initial=0.0, speed=0.2, parent=transform_section.content)
        self.scale_y_log2 = self.create_momentum_slider("Scale Y", initial=0.0, speed=0.2, parent=transform_section.content)
    
        self.rotation = self.create_rotation_control(parent=transform_section.content)
        self.flip_h = tk.IntVar()
        self.flip_v = tk.IntVar()
        tk.Checkbutton(transform_section.content, text="Flip Horizontally", variable=self.flip_h, command=self.redraw).pack(anchor="w")
        tk.Checkbutton(transform_section.content, text="Flip Vertically", variable=self.flip_v, command=self.redraw).pack(anchor="w")

        # Export
        tk.Button(
            self.control_frame,
            text="Export TSVs",
            command=self.export_transformed_data,
            bg="#007BFF",     # Bootstrap-style blue
            fg="white",       # White text for contrast
            activebackground="#0056b3",  # Darker blue on hover/click
            activeforeground="white"
        ).pack(pady=10, fill="x")

        tk.Button(
            self.control_frame,
            text="Export H5AD",
            bg="#007BFF",
            fg="white",
            command=self.export_to_h5ad,
            activebackground="#0056b3",  # Darker blue on hover/click
            activeforeground="white"
        ).pack(fill=tk.X, pady=5)



        self.anndata = None
        self.hires_image = None
        self.spots = None
        self.spot_ids = None
        self.tk_image = None
        self.spot_drawings = []

    def on_canvas_press(self, event):
        self._dragging = True
        self._drag_start = (event.x, event.y)

    def on_canvas_drag(self, event):
        if self._dragging:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._drag_start = (event.x, event.y)

            # Adjust shift_x and shift_y (in canvas units)
            self.shift_x.set(self.shift_x.get() + dx)
            self.shift_y.set(self.shift_y.get() + dy)
            self.redraw()

    def on_canvas_release(self, event):
        self._dragging = False

    def create_slider_with_entry(self, text, from_, to, initial, resolution=0.1, on_change=None, parent=None):
        parent = parent or self.control_frame
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        label = tk.Label(frame, text=text)
        label.pack(anchor="w")

        var = tk.DoubleVar(value=initial)

        def on_var_change(val=None):
            if on_change:
                on_change(var.get())
            self.redraw()

        slider = tk.Scale(frame, variable=var, from_=from_, to=to,
                        orient=tk.HORIZONTAL, resolution=resolution,
                        command=lambda e: on_var_change())
        slider.pack(fill=tk.X)

        entry = tk.Entry(frame, textvariable=var, width=6)
        entry.pack(anchor="e")
        entry.bind("<Return>", lambda e: on_var_change())

        return var

    def create_momentum_slider(self, text, initial=1.0, speed=0.1, parent=None):
        parent = parent or self.control_frame
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        label = tk.Label(frame, text=text)
        label.pack(anchor="w")

        value_var = tk.DoubleVar(value=initial)
        slider_var = tk.IntVar(value=0)  # Centered

        entry = tk.Entry(frame, textvariable=value_var, width=6)
        entry.pack(anchor="e")

        slider = tk.Scale(frame, from_=-100, to=100, variable=slider_var,
                        orient=tk.HORIZONTAL, showvalue=False)
        slider.pack(fill="x")

        running = {"active": False}

        def update_value():
            if not running["active"]:
                return
            delta = slider_var.get()
            if delta != 0:
                scale = (abs(delta) / 100) ** 1.5
                step = speed * scale * (1 if delta > 0 else -1)
                new_val = value_var.get() + step
                value_var.set(round(new_val, 4))  # round for clarity
                self.redraw()
            slider.after(50, update_value)

        def on_press(event):
            running["active"] = True
            update_value()

        def on_release(event):
            running["active"] = False
            slider_var.set(0)

        def on_entry(event=None):
            try:
                val = float(entry.get())
                value_var.set(val)
                self.redraw()
            except ValueError:
                pass  # ignore bad input

        slider.bind("<ButtonPress-1>", on_press)
        slider.bind("<ButtonRelease-1>", on_release)
        entry.bind("<Return>", on_entry)

        return value_var


    def create_rotation_control(self, parent=None):
        parent = parent or self.control_frame
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=5)

        label = tk.Label(frame, text="Rotation (°)")
        label.pack(anchor="w")

        self.rotation_var = tk.DoubleVar(value=0)

        entry = tk.Entry(frame, textvariable=self.rotation_var, width=6)
        entry.pack(anchor="e")
        entry.bind("<Return>", lambda e: self.redraw())

        btn_frame = tk.Frame(frame)
        btn_frame.pack()

        def rotate(delta):
            self.rotation_var.set(self.rotation_var.get() + delta)
            self.redraw()

        tk.Button(btn_frame, text="⟲ CCW", command=lambda: rotate(-90)).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="⟳ CW", command=lambda: rotate(90)).pack(side=tk.LEFT)

        return self.rotation_var

    def reset_transformations(self):
        self.shift_x.set(0)
        self.shift_y.set(0)
        self.spot_radius_multiplier_log2.set(0)
        self.scalef_multiplier_log2.set(0)
        self.scale_x_log2.set(0)
        self.scale_y_log2.set(0)
        self.flip_h.set(0)
        self.flip_v.set(0)
        self.rotation_var.set(0)
        self.spots_scaled = self.original_spots_scaled.copy()
        self.redraw()

    def load_anndata(self):
        file_path = filedialog.askopenfilename(filetypes=[("h5ad files", "*.h5ad")])
        if not file_path:
            return

        self.anndata = anndata.read_h5ad(file_path)

        if len(self.anndata.uns["spatial"]) == 1:
            self.lib_id = list(self.anndata.uns["spatial"].keys())[0]
        else:
            raise Exception('Check lib_id')

        # Get hires image and coordinates
        img_data = self.anndata.uns["spatial"]
        lib_id = list(img_data.keys())[0]
        image_info = img_data[lib_id]["images"]["hires"]
        # image_path = img_data[lib_id]["metadata"].get("source_image_path", None)

        # Load image as PIL.Image
        if isinstance(image_info, np.ndarray):
            image = Image.fromarray(image_info)
        # elif image_path:
        #     image = Image.open(image_path)
        else:
            raise ValueError("No valid hires image found.")

        self.original_image = image
        original_size = image.size

        # Resize for display
        image = self.resize_image(image)
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Compute scale factor for spot coordinate display
        self.display_scale = image.width / original_size[0]

        # Load and scale coordinates
        self.spots = self.anndata.obsm["spatial"]
        self.spots = self.spots[~np.isnan(self.spots).any(axis=1)]
        self.scalefactor = img_data[lib_id]["scalefactors"]["tissue_hires_scalef"]
        self.spots_scaled = self.spots * self.scalefactor * self.display_scale
        self.spot_radius = img_data[lib_id]["scalefactors"]["spot_diameter_fullres"] / 2

        self.original_spots_scaled = self.spots_scaled.copy()

        # Normalize spots to fit canvas if needed
        # canvas_w = self.tk_image.width()
        # canvas_h = self.tk_image.height()
        # if self.all_spots_outside_canvas(self.spots_scaled, canvas_w, canvas_h):
        #     print("[INFO] All spots outside canvas — normalizing to fit")
        #     self.spots_scaled = self.normalize_spots_to_canvas(
        #         self.spots_scaled, canvas_w, canvas_h
        #     )

        self.redraw()

    def resize_image(self, image):
            max_w, max_h = self.max_display_size
            w, h = image.size
            scale = min(max_w / w, max_h / h, 1.0)
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.LANCZOS)
    
    def all_spots_outside_canvas(self, coords, canvas_w, canvas_h, margin=0):
        x_valid = (coords[:, 0] >= -margin) & (coords[:, 0] <= canvas_w + margin)
        y_valid = (coords[:, 1] >= -margin) & (coords[:, 1] <= canvas_h + margin)
        inside = x_valid & y_valid
        return not np.any(inside)
    
    def normalize_spots_to_canvas(self, coords, canvas_width, canvas_height, padding=20):
        # 1. Get bounding box
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)

        # 2. Compute scale
        spot_width = max_x - min_x
        spot_height = max_y - min_y

        scale_x = (canvas_width - 2 * padding) / spot_width 
        scale_y = (canvas_height - 2 * padding) / spot_height
        scale = min(scale_x, scale_y)  # Uniform scale to preserve aspect

        # 3. Apply scale and shift
        coords_norm = (coords - [min_x, min_y]) * scale + [padding, padding]

        return coords_norm

    def transform_spots(self, spots, mode=None):
        spots = spots.copy()

        # Apply shift
        shift_x = self.shift_x.get()
        shift_y = self.shift_y.get()
        if mode == 'export':
            shift_x = shift_x / self.scalefactor / self.display_scale
            shift_y = shift_y / self.scalefactor / self.display_scale
        spots[:, 0] += shift_x
        spots[:, 1] += shift_y

        # Apply flip (screen-aligned)
        if self.flip_h.get():
            spots[:, 0] = 2 * np.mean(spots[:, 0]) - spots[:, 0]
        if self.flip_v.get():
            spots[:, 1] = 2 * np.mean(spots[:, 1]) - spots[:, 1]

        # Apply scale (screen-aligned)
        cx, cy = np.mean(spots[:, 0]), np.mean(spots[:, 1])
        spots[:, 0] = cx + (spots[:, 0] - cx) * 2**self.scale_x_log2.get()
        spots[:, 1] = cy + (spots[:, 1] - cy) * 2**self.scale_y_log2.get()

        # Now apply rotation (final)
        angle = np.deg2rad(self.rotation_var.get())
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ])

        centered = spots - [cx, cy]
        rotated = centered @ R.T
        transformed = rotated + [cx, cy]

        return transformed

    def draw_cluster_outlines(self, spots, min_samples=3):
        # Filter out NaNs
        valid = ~np.isnan(spots).any(axis=1)
        spots = spots[valid]
        if len(spots) < 3:
            return

        # Try auto-increasing EPS until we find at least one cluster
        max_eps = 200
        for eps in range(1, max_eps + 1, 5):
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(spots)
            labels = clustering.labels_
            if len(set(labels)) > 1 and any(l != -1 for l in labels):
                break  # Found valid clusters

        # self.cluster_eps.set(eps)  # Update slider if shown

        # Cluster with DBSCAN
        eps = self.shapely_alpha.get()
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(spots)
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:
                continue  # noise

            cluster_pts = spots[labels == label]
            if len(cluster_pts) < 3:
                continue

            shape = alpha_shape(cluster_pts, alpha=self.shapely_alpha.get())  # Adjust alpha as needed
            if not shape.is_empty:
                coords = list(mapping(shape)["coordinates"])
                for ring in coords:
                    flat = [coord for point in ring for coord in point]
                    self.canvas.create_polygon(*flat, outline="red", fill="", width=2)

    def redraw(self):
        self.canvas.delete("all")

        # Draw resized image
        self.tk_image = ImageTk.PhotoImage(self.resize_image(self.original_image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Draw image boundary
        self.canvas.create_rectangle(
            0, 0,
            self.tk_image.width(), self.tk_image.height(),
            outline="blue", width=2
        )

        # Transform and draw spots
        transformed_spots = self.transform_spots(self.spots_scaled) * 2**self.scalef_multiplier_log2.get()
        r = self.spot_radius * 2**self.spot_radius_multiplier_log2.get() * self.scalefactor * 2**self.scalef_multiplier_log2.get() * self.display_scale
        self.spot_drawings.clear()

        if self.show_spots.get():
            for x, y in transformed_spots:
                if not np.isnan(x) and not np.isnan(y):
                    oval = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                                fill="blue", outline="black")
                    self.spot_drawings.append(oval)

        # if self.show_clusters.get():
        #     self.draw_cluster_outlines(transformed_spots)

    def export_transformed_data(self):
        import os
        from tkinter import filedialog
        import json

        # Ask for save directory
        out_dir = filedialog.askdirectory(title="Choose export directory")
        if not out_dir:
            return

        # Create original + transformed coords
        full_spots = self.anndata.obsm["spatial"]
        valid_mask = ~np.isnan(full_spots).any(axis=1)
        transformed_coords = np.full_like(full_spots, np.nan)

        # Transform only valid spots
        valid_spots = full_spots[valid_mask]
        scalefactor_hires = self.scalefactor * 2**self.scalef_multiplier_log2.get()

        # Working but shift x shift y
        spots_scaled = valid_spots
        transformed = self.transform_spots(spots_scaled, mode='export')

        transformed_coords[valid_mask] = transformed

        # Get barcodes
        barcodes = self.anndata.obs_names

        # Save TSV
        tsv_path = os.path.join(out_dir, "transformed_coords.tsv")
        with open(tsv_path, "w") as f:
            f.write("barcode\tx\ty\n")
            for bc, coord in zip(barcodes, transformed_coords):
                x, y = coord
                x_str = "" if np.isnan(x) else f"{x:.2f}"
                y_str = "" if np.isnan(y) else f"{y:.2f}"
                f.write(f"{bc}\t{x_str}\t{y_str}\n")
        print(f"[INFO] Saved coordinates to: {tsv_path}")

        # Save scalefactors.json
        json_path = os.path.join(out_dir, "scalefactors_json.json")
        scalefactors = {
            "spot_diameter_fullres": 2 * self.spot_radius * 2**self.spot_radius_multiplier_log2.get(),
            "tissue_hires_scalef": float(scalefactor_hires),
            "tissue_lowres_scalef": 1.0  # You can change this if needed
        }
        with open(json_path, "w") as jf:
            json.dump(scalefactors, jf, indent=2)
        print(f"[INFO] Saved scalefactors to: {json_path}")


    def export_to_h5ad(self):
        import os
        from tkinter import filedialog
        import json
        import scanpy as sc

        if not hasattr(self, "anndata"):
            print("[ERROR] No AnnData object loaded.")
            return

        # Ask where to save the new h5ad
        save_path = filedialog.asksaveasfilename(
            title="Save transformed AnnData",
            defaultextension=".h5ad",
            filetypes=[("H5AD files", "*.h5ad")]
        )
        if not save_path:
            return

        # Extract original coords
        full_spots = self.anndata.obsm["spatial"]
        valid_mask = ~np.isnan(full_spots).any(axis=1)
        transformed_coords = np.full_like(full_spots, np.nan, dtype=np.float64)

        # Scale factor
        scalefactor_hires = self.scalefactor * 2**self.scalef_multiplier_log2.get()

        # Transform only valid spots
        spots_scaled = full_spots[valid_mask].astype(np.float64, copy=True)
        transformed = self.transform_spots(spots_scaled, mode='export')

        transformed_coords[valid_mask] = transformed

        # Update the AnnData object
        self.anndata.obsm["spatial"] = transformed_coords

        # Update scalefactors if stored in uns
        self.anndata.uns["spatial"][self.lib_id]["scalefactors"] = {
            "spot_diameter_fullres": float(2 * self.spot_radius * 2**self.spot_radius_multiplier_log2.get()),
            "tissue_hires_scalef": float(scalefactor_hires),
            "tissue_lowres_scalef": 1.0
        }

        # Save updated AnnData
        self.anndata.write_h5ad(save_path)
        print(f"[INFO] Saved updated h5ad: {save_path}")


def alpha_shape(points, alpha):
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 1e-10)
        circum_r = a * b * c / (4.0 * np.sqrt(area))
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    edge_points = [(points[i], points[j]) for i, j in edges]
    m = unary_union([LineString([p1, p2]) for p1, p2 in edge_points])
    triangles = list(polygonize(m))
    return unary_union(triangles)


if __name__ == "__main__":
    root = tk.Tk()
    app = SpotOverlayApp(root)
    root.mainloop()
