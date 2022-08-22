import mmcv
import numpy as np
from sklearn.model_selection import train_test_split
import funcy
from tabulate import tabulate
import coloredlogs, logging
import itertools, os, json, urllib.request
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


coloredlogs.install()
logging.basicConfig(format='[%(asctime)s : %(message)s %(filename)s]',
                    log_colors='green', loglevel=logging.ERROR)


def check_instances_categories(file, annotations, class_names):
    """
    #### category index should start from 1
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,))
    for anno in annotations:
        classes = np.asarray(
            [anno["category_id"] - 1]
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                    classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logging.basicConfig(format='[%(asctime)s : %(message)s %(filename)s]',
                        log_colors='green', loglevel=logging.INFO)

    logging.info('\n' + '\033[92m' + 'Categories and Instances in the ' + file + ':' + '\033[96m' + '\n' + table)


def save_coco(file, images, annotations, categories):
    check_instances_categories(file, annotations, [category['name'] for category in categories])
    with open(file, 'wt') as coco:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories}, coco, indent=2,
                  sort_keys=False)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def dataset_split(annotation_file, path_to_train, path_to_test, ratio):
    with open(annotation_file, 'rt') as annotations:
        coco = json.load(annotations)
        images = coco['images']

        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        train, test = train_test_split(images, train_size=ratio)

        save_coco(path_to_train, train, filter_annotations(annotations, train), categories)
        save_coco(path_to_test, test, filter_annotations(annotations, test), categories)


def check_download_images(imgs_info, path):
    download_error = {}
    for img_info in imgs_info:
        image_path = img_info['image_path']
        if isinstance(img_info['url'], str):
            image_url = [''.join(img_info['url'])]
        else:
            image_url = img_info['url']
        download_sucess = False
        f_path = os.path.abspath(os.path.dirname(image_path) + os.path.sep + ".")
        download_path = os.path.join(path, img_info['file_name'])
        if os.access(download_path, mode=os.R_OK):
            continue
        else:
            os.makedirs(f_path, exist_ok=True)
            for url in image_url:
                try:
                    urllib.request.urlretrieve(url, download_path)
                    download_sucess = True
                    break
                except Exception as e:
                    continue
            if download_sucess is False:
                download_error[img_info['file_name']] = image_path
                continue
        img = cv2.imread(image_path, -1)
        dim = (img.shape[1], img.shape[0])
        dim_origin = (img_info['width'], img_info['height'])
        if dim != dim_origin:
            img = cv2.resize(img, dim_origin, cv2.INTER_AREA)
            cv2.imwrite(image_path, img)
    images_with_expired_urls = list(download_error.values())
    if len(images_with_expired_urls) != 0:
        for img_dir in images_with_expired_urls:
            print('\n' + 'The image " ' + img_dir + ' " is not exist.')
        logging.info('\n' + 'You need to download those images by yourself to: ' + path + '\n')
    else:
        logging.info('\n' + 'All the needed images have been downloaded to: ' + path + '\n')


def check_anno_index(path_to_anno):
    with open(path_to_anno) as coco_format_anno:
        anno = json.load(coco_format_anno)
    annotations = anno['annotations']
    categories = anno['categories']
    index_start_zero = False
    if categories[0]['id'] != 0:
        return index_start_zero, anno
    else:
        index_start_zero = True
        for category in categories:
            category['id'] += 1
        for annotation in annotations:
            annotation['category_id'] += 1
    anno_sorted_index = {
        "images": anno['images'],
        "annotations": annotations,
        "categories": categories
    }
    return index_start_zero, anno_sorted_index

def binarize(T, nb_classes):
    import torch
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def pairwise_mahalanobis(X, means, log_vars):
    """
    Computes pairwise squared Mahalanobis distances between X (data points) and a set of distributions
    :param X: [N, F] where N is the batch size and F is the feature dimension
    :param means: [C, F] C is the number of classes
    :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    :return: pairwise squared Mahalanobis distances... [N, C, F] matrix
    i.e., M_ij = (x_i-means_j)\top * inv_cov_j * (x_i - means_j)

    """
    sz_batch = X.size(0)
    nb_classes = means.size(0)

    new_X = torch.unsqueeze(X, dim=1)  # [N, 1, F]
    new_X = new_X.expand(-1, nb_classes, -1)  # [N, C, F]

    new_means = torch.unsqueeze(means, dim=0)  # [1, C, F]
    new_means = new_means.expand(sz_batch, -1, -1)  # [N, C, F]

    # pairwise distances

    diff = new_X - new_means # [N, C, F]

    # convert log_var to covariance
    # TODO: why log_vars here can be considered as the log of the covariance matrix?
    covs = torch.unsqueeze(torch.exp(log_vars), dim=0)  # [1, C, F]

    # the squared Mahalanobis distances
    # TODO: why calculate in this way?
    # calculate the squared value, and then divide by the covariance
    # :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    M = torch.div(diff.pow(2), covs).sum(dim=-1)  # [N, C]

    return M

# Class Distributions to Hypergraph
class CDs2Hg(nn.Module):
    def __init__(self, nb_classes, sz_embed, tau=32, alpha=0.9):
        super(CDs2Hg, self).__init__()
        # Parameters (means and covariance)
        self.means = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda()) # [C, F]
        self.log_vars = nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda()) # [C, F]

        # Initialization
        nn.init.kaiming_normal_(self.means, mode='fan_out')
        nn.init.kaiming_normal_(self.log_vars, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.tau = tau
        self.alpha = alpha

    def forward(self, X, T):
        mu = self.means
        log_vars = self.log_vars
        # ReLU with upper bound 6
        log_vars = F.relu6(log_vars)

        # L2 normalize
        X = F.normalize(X, p=2, dim=-1) # [N, F]
        mu = F.normalize(mu, p=2, dim=-1)

        # Labels of each distributions (NxC matrix)
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)

        # Compute pairwise mahalanobis distances (NxC matrix)
        distance = pairwise_mahalanobis(X, mu, log_vars)

        # Distribution loss
        # equation 2
        mat = F.softmax(-1 * self.tau * distance, dim=1) # [N, C]
        # equation 3
        P_one_hot = P_one_hot.detach().to("cuda:{}".format(mat.get_device()))
        loss = torch.sum(mat * P_one_hot, dim=1)
        non_zero = loss != 0
        loss = -torch.log(loss[non_zero])

        # Hypergraph construction
        # P_one_hot.sum(dim=0) calculates the number of examples in each class
        # if P_one_hot = [
        # [1,0,0,0],
        # [0,1,0,0],
        # [0,1,0,0],
        # [0,0,0,1]
        # ]
        # e.g. class_within_batch = tensor([0, 1, 3]), 0 1 3 are subscripts of the classes
        class_within_batch = torch.nonzero(
            P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        # distance[:, class_within_batch] is the pairwise mahalanobis distances of the classes, the classes
        # are which with trurhy labels and existed in the one hot matrix P_one_hot
        exp_term = torch.exp(-1 * self.alpha * distance[:, class_within_batch])

        # exp_term * (1 - P_one_hot[:, class_within_batch]) to get the distance between the classes,
        # which are not labelled as true
        # then plus P_one_hot[:, class_within_batch],
        # get the classes distance for labelled classes und unlabelled classes
        # if P_one_hot = [
        # [1,0,0,0],
        # [0,1,0,0],
        # [0,1,0,0],
        # [0,0,0,1]
        # ]
        # if exp_term = [
        # [0.1,0.2,0.3,0.4],
        # [0.5,0.6,0.7,0.8],
        # [0.9,0.2,0.3,0.4],
        # [0.5,0.6,0.7,0.8]
        # ]
        # equation 4 for S_{ij}, each line reprents the distance between the classes, actually one semantic tuple
        H = P_one_hot[:, class_within_batch] + exp_term * \
            (1 - P_one_hot[:, class_within_batch])

        return loss.mean(), H


# Hypergraph Neural Networks (AAAI 2019)
class HGNN(nn.Module):
    def __init__(self, nb_classes, sz_embed, hidden):
        super(HGNN, self).__init__()

        self.theta1 = nn.Linear(sz_embed, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lrelu = nn.LeakyReLU(0.1)

        self.theta2 = nn.Linear(hidden, nb_classes)

    def compute_G(self, H):
        # the number of hyperedge --> number of classes in this batch
        n_edge = H.size(1) # --> C
        # the weight of the hyperedge
        we = torch.ones(n_edge).cuda() # --> [C]
        # the degree of the node
        we = we.detach().to("cuda:{}".format(H.get_device()))
        Dv = (H * we).sum(dim=1) # --> [C]
        # the degree of the hyperedge
        De = H.sum(dim=0) # --> [N]

        We = torch.diag(we)
        # returns a 1-D tensor with the diagonal elements of Dv^-0.5, size is min(N, C)
        inv_Dv_half = torch.diag(torch.pow(Dv, -0.5))
        inv_De = torch.diag(torch.pow(De, -1))
        H_T = torch.t(H)

        # propagation matrix
        # ! in detail need to read the paper AAAI 2019: Hypergraph Neural Networks
        G = torch.chain_matmul(inv_Dv_half, H, We, inv_De, H_T, inv_Dv_half)

        return G

    def forward(self, X, H):
        G = self.compute_G(H)

        # 1st layer
        X = G.matmul(self.theta1(X))
        X = self.bn1(X)
        X = self.lrelu(X)

        # 2nd layer
        out = G.matmul(self.theta2(X))

        return out