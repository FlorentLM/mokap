import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


class SkeletonViewer:

    def __init__(self, soup, candidates, bones_def):
        self.soup = soup

        self.skel_graph = nx.Graph()
        if bones_def and isinstance(bones_def[0][0], int):
            for u_idx, v_idx in bones_def:
                try:
                    u = soup.keypoint_names[u_idx]
                    v = soup.keypoint_names[v_idx]
                    self.skel_graph.add_edge(u, v)
                except:
                    pass
        else:
            self.skel_graph.add_edges_from(bones_def)

        self.candidates = candidates
        self.frames = sorted(self.candidates.keys())

        if not self.frames:
            print("No candidates to show.")
            return

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.1)

        self.slider = Slider(plt.axes([0.2, 0.02, 0.6, 0.03]), 'Frame', 0, len(self.frames) - 1, valinit=0, valfmt='%d')
        self.slider.on_changed(self.update)

        self.draw(0)
        plt.show()

    def update(self, val):
        self.draw(int(self.slider.val))

    def draw(self, idx):
        self.ax.clear()
        f_idx = self.frames[idx]
        self.ax.set_title(f"Frame {f_idx}")

        # Draw soup
        try:
            s = self.soup.get_frame(f_idx)
            if s.num_points > 0:
                self.ax.scatter(s.positions[:, 0], s.positions[:, 1], s.positions[:, 2], c='k', alpha=0.1, s=5)
        except:
            pass

        # Draw candidates
        cands = self.candidates[f_idx]
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(cands))))

        for i, c in enumerate(cands):
            kps = c.keypoints

            xyz = np.array(list(kps.values()))
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colors[i], s=50, edgecolors='w')

            for u, v in self.skel_graph.edges():
                if u in kps and v in kps:
                    p1, p2 = kps[u], kps[v]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=colors[i], lw=2)

            cent = np.nanmean(xyz, axis=0)
            self.ax.text(cent[0], cent[1], cent[2], f"S:{c.scale:.2f}", color='k', fontsize=8, weight='bold')

        if len(cands) > 0:
            # Auto-center on first candidate
            vals = np.array(list(cands[0].keypoints.values()))
            c = np.mean(vals, axis=0)
            r = 15
            self.ax.set_xlim(c[0] - r, c[0] + r)
            self.ax.set_ylim(c[1] - r, c[1] + r)
            self.ax.set_zlim(c[2] - r, c[2] + r)