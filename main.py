import gpxpy
import imageio
import matplotlib.pyplot as plt
import itertools
import networkx as nx
import osmnx as ox

DEBUG = False
g = ox.graph_from_place("Stamford, CT", network_type="drive")
street_names = {
    "Ralsey Road",
    "Ralsey Road South",
    "Sagamore Road",
    "Woolsey Road",
    "Ocean Drive West",
    "Stamford Avenue",
    "Shippan Avenue",
    "Westcott Road",
    "Sea Beach Drive",
    "Ocean Drive East",
    "Rockledge Drive",
    "Hobson Street",
    "Fairview Avenue",
    "Brightside Drive",
    "Sound Avenue",
    "Cresthill Place",
    "Van Rensselaer Avenue",
    "Verplank Avenue",
    "Westminster Road",
    "Miramar Road",
    "Saddle Rock Road",
    "Rogers Road",
    "Lighthouse Way",
}

selected_nodes = []

for u, v, e in g.edges(data=True):
    name = e.get("name", [])
    check_names = name if isinstance(name, list) else [name]

    for n in check_names:
        if n in street_names:
            selected_nodes.append(u)
            selected_nodes.append(v)

# Remove nodes on Shippan avenue above certain level
selected_nodes = [
    u for u in selected_nodes if g.nodes[u]["y"] < 41.03356
]  # hard code y coordinate to stop at shippan ave and Ralsey
# selected_nodes = [u for u in selected_nodes if 'highway' not in g.nodes[u]]


g_proj = ox.project_graph(g.subgraph(selected_nodes))
g = ox.consolidate_intersections(
    g_proj, rebuild_graph=True, tolerance=15, dead_ends=False
)


# Computing the shortest cycle

# downgrade to a normal undirected graph
ug = nx.MultiGraph(g)

# extract odd degree nodes
odd_deg_nodes = [g for g, v in ug.degree if v % 2 != 0]

# find sets of pairings
odd_node_pairs = list(itertools.combinations(odd_deg_nodes, 2))

# Compute shortest paths
odd_edge_graph = nx.Graph()
dist_mat = {}
shortest_path_mat = {}
for i in range(len(odd_deg_nodes)):
    for j in range(i, len(odd_deg_nodes)):
        u = odd_deg_nodes[i]
        v = odd_deg_nodes[j]

        # compute distances and paths
        min_dist = nx.shortest_path_length(ug, u, v, weight="length")
        min_path = nx.shortest_path(ug, u, v, weight="length")

        # weight is negative bc we will be using "max weight matching"
        odd_edge_graph.add_edge(u, v, weight=-min_dist)

        dist_mat[(u, v)] = min_dist
        dist_mat[(v, u)] = min_dist

        shortest_path_mat[(u, v)] = min_path
        shortest_path_mat[(v, u)] = min_path[::-1]


mates = nx.algorithms.max_weight_matching(odd_edge_graph, maxcardinality=True)


# Add matches to original graph, ug
eulerian_ug = ug.copy()

if DEBUG:
    x_to_index = {v: k for k, v in eulerian_ug.nodes(data="x")}

    def add_node_labels(g, ax):
        node_df, edge_df = ox.graph_to_gdfs(g)

        for i, node in node_df.iterrows():
            text = str(x_to_index[node.x])
            ax.annotate(text, (node.x, node.y), c="white")

    fig, ax = ox.plot_graph(eulerian_ug, show=False)
    add_node_labels(eulerian_ug, ax)
    fig.savefig("pre-add.png")


for v1, v2 in list(mates):
    eulerian_ug.add_edge(v1, v2, weight=dist_mat[(v1, v2)], length=dist_mat[(v1, v2)])

if DEBUG:
    fig, ax = ox.plot_graph(eulerian_ug, show=False)
    add_node_labels(eulerian_ug, ax)

    fig.savefig("post-add.png")

# Compute eulerian circuit

route = []
edge_route = []
for edge in nx.eulerian_circuit(eulerian_ug, source=47, keys=True):  # 47 is my house
    u, v, _ = edge

    # replace with the shortest path IFF edge doesn't exist in real graph:
    to_add = [u, v] if ug.has_edge(u, v) else shortest_path_mat[(u, v)]
    route += to_add

# DEDUP the route (there are repeated nodes)
new_route = [route[0]]
for v in route[1:]:
    if new_route[-1] != v:
        new_route.append(v)

# remove added edges
for v1, v2 in list(mates):
    eulerian_ug.remove_edge(v1, v2)


def print_route(g, route):
    print("\n\n\n###### ROUTE #######")
    last = route[0]
    for u in route[1:]:
        temp = g.edges[last, u, 0]
        print(
            f"from {last} to {u} take {temp['name']} for {round(temp['length'])} meters"
        )
        last = u


if DEBUG:
    print_route(eulerian_ug, new_route)

# compute total distance
dist = 0
last = new_route[0]
for u in new_route[1:]:
    dist += eulerian_ug.edges[last, u, 0]["length"]
    last = u

print("final distance in km: ", round(dist / 1000, 2))


def plot_route_at_step(
    g,
    visited,
    node_labels=True,
    edge_labels=True,
    output_file="temp.png",
    dist=None,
    redund_dist=None,
):
    color_map = {0: "white", 1: "yellow", 2: "red", 3: "orange"}
    ecolors = [
        color_map[
            visited.get(
                (
                    u,
                    v,
                ),
                0,
            )
        ]
        for u, v in g.edges()
    ]

    fig, ax = ox.plot_graph(
        g, edge_color=ecolors, show=False, bgcolor="#7B7A7A", edge_linewidth=1.5
    )

    node_df, edge_df = ox.graph_to_gdfs(g)
    if node_labels:
        for _, node in node_df.iterrows():
            ax.annotate(str(x_to_index[node.x]), (node.x, node.y), c="white")

    if edge_labels:
        labeled_edges = set()
        for _, edge in (
            edge_df.fillna("").sort_values("length", ascending=False).iterrows()
        ):
            text = edge["name"]
            if isinstance(text, list):
                text = list[0]
            if text not in labeled_edges:
                c = edge["geometry"].centroid

                ax.annotate(text, (c.x - 100, c.y), c="white", fontsize=8)
                labeled_edges.add(text)

    if dist:
        ax.annotate(
            f"Total distance: {round(running_dist/1000, 2)} km",
            (0.58, 0.85),
            c="white",
            xycoords="figure fraction",
        )

    if redund_dist:
        ax.annotate(
            f"Total duplicated: {round(redund_dist/1000, 2)} km",
            (0.58, 0.83),
            c="white",
            xycoords="figure fraction",
        )

    # add legend
    ax.plot([], [], color="white", label="Unvisited")
    ax.plot([], [], color="yellow", label="One Visit")
    ax.plot([], [], color="red", label="Two Visits")

    # Add a legend
    leg = ax.legend(loc="lower right")
    leg.get_frame().set_facecolor("#7B7A7A")
    leg.set_frame_on(False)
    for text in leg.get_texts():
        text.set_color("white")

    fig.savefig(output_file)
    plt.close()


visited = {}
last = new_route[0]
running_dist = 0
redund_dist = 0
f_names = []
for i, v in enumerate(new_route[1:]):
    output_file = f"./slides/slide_{i}.png"
    edge = (last, v) if last < v else (v, last)

    visited[edge] = visited.get(edge, 0) + 1
    running_dist += eulerian_ug.edges[last, v, 0]["length"]
    if visited.get(edge, 0) > 1:
        redund_dist += eulerian_ug.edges[last, v, 0]["length"]
    plot_route_at_step(
        eulerian_ug,
        visited,
        output_file=output_file,
        edge_labels=False,
        dist=running_dist,
        node_labels=False,
        redund_dist=redund_dist,
    )
    last = v
    f_names.append(output_file)

print("total duplicated distance", round(redund_dist, 2))

# Animating the route by concatenating all of our outputfiles
with imageio.get_writer("animation.gif", mode="I", duration=0.25) as writer:
    for image_file in f_names:
        image = imageio.imread(image_file)
        writer.append_data(image)


gpx = gpxpy.gpx.GPX()

gpx_track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(gpx_track)

gpx_segment = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment)

# convert new route to edges:

for i in range(len(new_route) - 1):
    u, v = new_route[i], new_route[i + 1]
    geom = eulerian_ug.edges[u, v, 0]["geometry"]
    proj_geom, _ = ox.projection.project_geometry(
        # geom, crs=g_proj.graph["crs"], to_latlong=True
        geom,
        crs=eulerian_ug.graph["crs"],
        to_latlong=True,
    )
    for lon, lat in proj_geom.coords:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=1))

with open("./shippaniltonian.gpx", "w+") as f:
    print(gpx.to_xml(), file=f)
