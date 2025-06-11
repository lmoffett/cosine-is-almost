# Code adapted from:
# Proto-Med EEG Paper codebase
import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import torch
from annoy import AnnoyIndex
from tqdm.auto import tqdm

from . import datasets

log = logging.getLogger(__name__)


def feature_extract(
    dataloader,
    model_path,
    save_dir,
    sample: int = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Extracts features from image data using a pre-trained model and saves the results.

    Saves the following:
        - labels: The labels corresponding to the extracted features and predictions.
        - extracted_features: The extracted features from the EEG data.

    Parameters:
    -----------
        dataloader (dataloader obj): A dataloader containing the image and label data
        model_path (str): File path to the pre-trained model.
        save_dir (str): Directory path to save the extracted features.
        short_data: only run pacmap on1 batch of data

    Returns:
    --------
        None

    Credit: Function mostly adopted from Proto-Med EEG codebase
    """
    ppnet = torch.load(model_path, map_location=device)

    extracted_features = []
    labels = []
    for batch_data_dict in dataloader:
        image = batch_data_dict["img"].to(device)
        label = batch_data_dict["target"]

        features = ppnet.backbone(image)
        features = ppnet.add_on_layers(features).detach().cpu().numpy()

        extracted_features.append(features)
        labels.append(label)

    extracted_features = np.concatenate(extracted_features, axis=0)
    labels = np.concatenate(labels, axis=0)
    log.info(f"Extracted features with shape {extracted_features.shape}")

    if sample:
        extracted_features = extracted_features[:sample]
        labels = labels[:sample]

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "labels.npy", labels)
    np.save(save_dir / "extracted_features.npy", extracted_features)


def build_annoy(
    feats,
    annoy_path,
    save=True,
):
    """
    Builds an AnnoyIndex for the given features and saves it to a specified path.

    Parameters:
    -----------
        feats (numpy.ndarray): The input features with shape (n_samples, n_features).
        annoy_path (str): The path to save the AnnoyIndex.
        save (bool, optional): Whether to save the AnnoyIndex to the specified path. Defaults to True.

    Returns:
    --------
        AnnoyIndex: The built AnnoyIndex.

    Credit: Function mostly adopted from Proto-Med EEG codebase
    """
    f = feats.shape[1]
    a = AnnoyIndex(f, "angular")

    log.info("Building AnnoyIndex for warm up feats of shape {}".format(feats.shape))
    a.set_seed(1)
    for idx, feat in enumerate(tqdm(feats)):
        assert (
            len(feat) == f
        ), f"Expected feat to have {f} dimensions, but got {len(feat)}"
        log.debug(f"Adding item {idx} to AnnoyIndex ({feat})")
        a.add_item(idx, feat)
        log.debug(f"Added item {idx}")

    a.build(1000, n_jobs=min(multiprocessing.cpu_count() - 2, 8))  # 10 trees

    if save:
        a.save(annoy_path)
        log.info(f"Annoy saved to {annoy_path}")

    return a


def annoy_pacmap_prepare_child(args):
    """
    Retrieves nearest neighbors using the AnnoyIndex for a specific item.

    Parameters:
    -----------
        args (list): A list containing the arguments idx_t, annoy_index_path, and f.

    Returns:
    --------
        tuple: A tuple containing the idx_t and the nearest neighbors.
    """
    idx_t, annoy_index_path, f = args
    annoy_index = AnnoyIndex(f, "angular")
    annoy_index.load(annoy_index_path)
    return idx_t, annoy_index.get_nns_by_item(idx_t, 1000 + 1)


def annoy_pacmap_prepare(feats, annoy_index, save_dir, num_cpus):
    """
    Prepares the neighbors for PaCMAP using the AnnoyIndex.

    Parameters:
    -----------
        feats (numpy.ndarray): The input features with shape (n_samples, n_features).
        annoy_index (AnnoyIndex): The AnnoyIndex used to retrieve nearest neighbors.
        save_dir (str): The directory path for saving temporary files.

    Returns:
    --------
        numpy.ndarray: The nearest neighbors for each item in feats.
    """
    log.info("Running")

    tmp_annoy_index_path = f"{save_dir}/tmp_annoy_index.ann"
    f = feats.shape[1]
    annoy_index.save(tmp_annoy_index_path)

    pool_args = []
    for idx_t in tqdm(range(feats.shape[0])):
        pool_args.append([idx_t, tmp_annoy_index_path, f])

    n, _ = feats.shape
    nbrs = np.zeros((n, min(n - 1, 1000)), dtype=np.int32)

    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(pool.imap(annoy_pacmap_prepare_child, pool_args), total=len(pool_args))
        )

    for rst in results:
        nbrs[rst[0], :] = rst[1][1:]

    pool.terminate()

    return nbrs


def save_pacmap_image(
    X_transformed,
    labels,
    save_dir,
    image_name,
    num_classes,
    plot_shape=(1, 1),
    divisor=1,
    class_labels=None,
):
    """
    Saves the PaCMAP image plot.

    Parameters:
    -----------
        X_transformed (numpy.ndarray): The transformed features with shape (n_samples, 2).
        labels (numpy.ndarray): The labels for each sample + each prototype appended at the end
        save_dir (str): The directory path for saving the image.
        image_name (str): The name of the image file.
        num_classes (int): Total number of classes to plot
        plot_shape ( (int,int) tuple ), dimensions of subplots
        divisor (int): number of classes the total classes are divided into
        class_labels (dict): a dict of {'label':idx} from the dataset obj

    Returns:
    --------
        Nothing
    """

    def clean_string(s):
        return "".join(
            c for c in s.replace(".", "").replace("_", " ") if not c.isdigit()
        )

    log.info("Plotting")

    # Validate that the product of plot dimensions equals the quotient
    assert plot_shape[0] * plot_shape[1] == divisor, (
        f"Plot configuration mismatch!\n"
        f"but plot_shape[0] * plot_shape[1] = {plot_shape[0] * plot_shape[1]}\n"
        f"The product of plot dimensions must equal divisor"
    )

    # define the plot shape
    R = plot_shape[0]
    C = plot_shape[1]

    fig, ax = plt.subplots(R, C, figsize=(16.5 * C, 10 * R))
    ax = np.array(ax).reshape(-1)

    plt.subplots_adjust(
        wspace=1.1,  # Increase horizontal space between subplots (default is 0.2)
        hspace=0.2,  # Increase vertical space between subplots (default is 0.2)
    )

    num_classes_per_plot = int(num_classes / divisor)
    class_labels = {v: clean_string(k) for k, v in class_labels.items()}

    if num_classes_per_plot <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 20))
        colors = np.concatenate([colors, colors], axis=0)

    elif num_classes_per_plot <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = np.concatenate([colors, colors], axis=0)

    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_classes_per_plot))
        colors = np.concatenate([colors, colors], axis=0)

    # init the values of the buffers
    proto_buffer = num_classes
    buffer = 0

    total_plot = 0

    for k in range(divisor):
        if class_labels:
            label_names = []
            for idx in range(num_classes_per_plot):
                label_names.append(class_labels[idx + buffer])

            label_names += [f"{name} Prototype" for name in label_names]

        else:
            label_names = [f"class {i + buffer}" for i in range(num_classes_per_plot)]
            label_names += [
                f"prototype of {i + buffer}" for i in range(num_classes_per_plot)
            ]

        labels_of_interest = [i + buffer for i in range(num_classes_per_plot)]
        labels_of_interest += [
            i + buffer + proto_buffer for i in range(num_classes_per_plot)
        ]

        sizes = [10 for i in range(num_classes_per_plot)]
        sizes += [300 for i in range(num_classes_per_plot)]

        for i in range(len(label_names)):
            if "Prototype" in label_names[i]:
                marker = "o"
                alpha_value = 0.3
            else:
                marker = "x"
                alpha_value = 0.4

            ax[total_plot].scatter(
                X_transformed[:, 0][labels == labels_of_interest[i]],
                X_transformed[:, 1][labels == labels_of_interest[i]],
                s=sizes[i],
                alpha=alpha_value,
                color=colors[i],
                marker=marker,
                label=label_names[i],
            )

        ax[total_plot].legend(
            fontsize=12,
            bbox_to_anchor=(1.05, 0.5),  # (1.05, 1) places it right outside
            loc="center left",
            ncol=2,
        )  # anchors the legend at its upper left corner

        ax[total_plot].set_title(
            f"PacMAP for classes {buffer}-{buffer+num_classes_per_plot}"
        )

        # plot everything that WASN'T in the labels as a gray 'x' #####
        ax[total_plot].scatter(
            X_transformed[:, 0][~np.isin(labels, labels_of_interest)],
            X_transformed[:, 1][~np.isin(labels, labels_of_interest)],
            s=10,  # You'll need to specify this
            alpha=0.05,
            color="gray",  # Or whatever color you want for "other" points
            marker="x",
            label="other classes",
        )

        total_plot += 1
        buffer += num_classes_per_plot

    output_image_path = Path(save_dir, image_name)
    plt.savefig(output_image_path, bbox_inches="tight")


def default_feature_reshape(features):
    """
    input shape is [bsz, feats, H, C]

    returns [bsz*H*C, feats]

    """
    features = np.transpose(features, (0, 2, 3, 1))
    features = features.reshape((-1, features.shape[-1]))

    return features


def dropout_reshape(features):
    """
    Reshape features with selective dropout per batch item.

    Args:
        features: numpy array of shape [bsz, feats, H, C]
        drops_per_img: number of (H,C) locations to drop per batch item
    """
    bsz, feats, H, C = features.shape
    drops_per_img = int(0.9 * H * C)

    # Transpose to [bsz, H, C, feats]
    features = np.transpose(features, (0, 2, 3, 1))

    processed_batches = []

    for batch_idx in range(bsz):
        # Shape: [H, C, feats]
        curr_features = features[batch_idx]

        # Shape: [H, C]
        mask = np.ones((H, C), dtype=bool)

        # Randomly select positions to drop
        drop_indices = np.random.choice(H * C, size=drops_per_img, replace=False)
        drop_h = drop_indices // C
        drop_c = drop_indices % C
        mask[drop_h, drop_c] = False

        # When we do curr_features[mask]:
        # 1. The mask (H, C) is flattened to (H*C,)
        # 2. curr_features is viewed as ((H*C), feats)
        # 3. We keep only the rows where mask is True
        kept_features = curr_features[mask]

        processed_batches.append(kept_features)

    # Concatenate along first dimension
    features = np.concatenate(processed_batches, axis=0)
    return features


def generate_pacmap(
    model_path,
    save_dir,
    image_name,
    num_cpus,
    reshape_function=default_feature_reshape,
    pacmap_distance_metric="angular",
    n_neighbors=50,
    built_annoy=True,
    plot_shape=(1, 1),
    divisor=1,
    class_labels=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Generates the PaCMAP visualization.

        Parameters:
        -----------
        model_path (str): The path to the model.
        save_dir (str): The directory path for saving the features.
        image_name (str): The name of the image file.
        reshape_function (function): A function that takes inputs 'x' and returns a reshaped , 2-dimension 'x'
        pacmap_distance_metric (str, optional): The distance metric for PaCMAP. Defaults to "angular".
        n_neighbors (int, optional): The number of neighbors for PaCMAP. Defaults to 50.
        built_annoy (bool, optional): Whether to build the AnnoyIndex. Defaults to True.
        plot_shape ( (int,int) tuple ), dimensions of subplots.
        divisor (int): number of classes the total classes are divided into.
        class_labels (dict): a dict of {'label':idx} from the dataset obj
        sample (int): number of images to sample to run pacmap on
    """
    plot_shape = (int(plot_shape[0]), int(plot_shape[1]))

    log.debug("Loading model")

    # no forward inference, so just load on CPU
    ppnet = torch.load(model_path, map_location=device)
    num_classes = ppnet.prototype_prediction_head.prototype_class_identity.shape[1]

    prototype_vectors = ppnet.prototype_layer.prototype_tensors.squeeze()
    prototype_vectors = prototype_vectors.detach().cpu().numpy()

    log.debug(f"Prototype vectors shape: {prototype_vectors.shape}")

    labels = np.load(save_dir / "labels.npy")
    num_classes = len(set(labels))
    features = np.load(save_dir / "extracted_features.npy")
    num_images = features.shape[0]

    log.debug(f"Original features shape: {features.shape}")

    log.debug(f"pre reshape features size: {features.shape}")
    features = reshape_function(features)

    log.debug(f"post reshape features size: {features.shape}")

    log.debug(f"New features shape: {features.shape}")
    log.debug(f"New labels shape: {labels.shape}")
    log.info(f"prototype_vectors shape: {prototype_vectors.shape}")

    log.debug(f"Final features shape (prototype vectors appended): {features.shape}")

    new_labels = []
    num_repeats = int(features.shape[0] / num_images)

    for i in range(len(labels)):
        new_labels += [labels[i] for _ in range(num_repeats)]

    proto_labels = torch.argmax(
        ppnet.prototype_prediction_head.prototype_class_identity, dim=1
    )

    labels = np.array(new_labels)
    proto_labels = proto_labels.cpu().numpy()

    plot_labels = np.concatenate((labels, proto_labels + num_classes), axis=0)
    np.save(save_dir / "labels_for_plotting.npy", plot_labels)

    # prototype vectors don't need reshaping
    features = np.concatenate((features, prototype_vectors), axis=0)

    if not built_annoy:
        log.debug("Building annoy")
        annoy_index = build_annoy(features, f"{save_dir}/annoy_index.ann", save=True)

        log.debug("Calculating neighbours")

        nbrs = annoy_pacmap_prepare(features, annoy_index, save_dir, num_cpus)

        np.save(save_dir / "nbrs.npy", nbrs)

    else:
        log.debug("Loading neighbours")

        nbrs = np.load(save_dir / "nbrs.npy")

    n, dim = features.shape

    scaled_dist = np.ones((n, n_neighbors))  # No scaling is needed
    scaled_dist = scaled_dist.astype(np.float32)

    log.debug("Calculating pair neighbours")

    pair_neighbors = pacmap.sample_neighbors_pair(
        features.astype(np.float32), scaled_dist, nbrs, np.int32(n_neighbors)
    )

    log.debug("Running PaCMAP")

    embedding = pacmap.PaCMAP(
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        pair_neighbors=pair_neighbors,
        distance=pacmap_distance_metric,
    )

    X_transformed = embedding.fit_transform(features, init="pca")

    log.debug("X_transformed shape: %s", X_transformed.shape)

    log.debug("Saving X_transformed")
    np.save(save_dir / "X_transformed.npy", X_transformed)

    save_pacmap_image(
        X_transformed,
        plot_labels,
        save_dir,
        image_name,
        num_classes,
        plot_shape,
        divisor,
        class_labels,
    )

    log.info("PaCMAP plotted")


def parse_plot_shape(ctx, param, value) -> Tuple[int, int]:
    """Convert plot shape string arguments to tuple of ints."""
    if not value:
        return (1, 1)
    try:
        return tuple(map(int, value))
    except ValueError:
        raise click.BadParameter("Plot shape must be two integers")


@click.command("pacmap")
@click.option("--dataset", type=str, help="Dataset to use")
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model",
)
@click.option(
    "--save-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save dir (holding features)",
)
@click.option(
    "--image-name", type=str, default="pacmap.png", help="Name of the image to save"
)
@click.option(
    "--plot-shape",
    callback=parse_plot_shape,
    nargs=2,
    type=int,
    default=(1, 1),
    help="Shape of plot as two integers (rows cols)",
)
@click.option(
    "--divisor",
    type=int,
    default=1,
    help="What to divide the total number of classes by",
)
@click.option(
    "--reshape-function",
    type=click.Choice(["default", "dropout"], case_sensitive=False),
    default="dropout",
    help="Name of reshape function for features",
)
@click.option(
    "--pacmap-distance-metric",
    type=str,
    default="angular",
    help="Distance metric for PaCMAP",
)
@click.option(
    "--n-neighbors", type=int, default=60, help="Number of neighbours for PaCMAP"
)
@click.option("--built-annoy", is_flag=True, help="Whether to build annoy index")
@click.option("--sample", type=int, help="Number of images to sample for the pacmap")
@click.option(
    "--device",
    type=str,
    default=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use (cuda/cpu)",
)
@click.option(
    "--cpu-count",
    type=int,
    default=lambda: multiprocessing.cpu_count(),
    help="Number of CPUs to use for PaCMAP",
)
def run(
    dataset: Optional[str],
    model_path: Path,
    save_dir: Path,
    image_name: str,
    plot_shape: Tuple[int, int],
    divisor: int,
    reshape_function: str,
    pacmap_distance_metric: str,
    n_neighbors: int,
    built_annoy: bool,
    sample: Optional[int],
    device: str,
    cpu_count: int,
):
    """Generate PaCMAP visualization of model features."""

    # Convert device string to torch.device
    device = torch.device(device)

    batch_sizes = {"train": 32, "project": 32, "val": 32}
    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    try:
        class_labels = split_dataloaders.train_loader.dataset.class_to_idx
    except AttributeError:
        # FIXME - probably want an explicit requirement of our datasets to return a listing of classes
        unique_classes = set()
        for item in split_dataloaders.train_loader.dataset:
            label = item["target"]
            unique_classes.add(label)
        class_labels = {str(label): label for label in unique_classes}

    if "default" in reshape_function:
        reshape_fct = default_feature_reshape
    if "dropout" in reshape_function:
        reshape_fct = dropout_reshape

    feature_extract(
        split_dataloaders.train_loader,
        model_path,
        save_dir,
        sample=sample,
        device=device,
    )
    generate_pacmap(
        model_path,
        save_dir,
        image_name,
        num_cpus=cpu_count,
        plot_shape=plot_shape,
        divisor=divisor,
        reshape_function=reshape_fct,
        pacmap_distance_metric=pacmap_distance_metric,
        n_neighbors=n_neighbors,
        built_annoy=built_annoy,
        class_labels=class_labels,
        device=device,
    )
