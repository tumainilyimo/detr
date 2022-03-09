import io

import matplotlib.pyplot as plt
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]


@st.cache(allow_output_mutation=True)
def get_hf_components(model_name_or_path):
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name_or_path)
    model = DetrForObjectDetection.from_pretrained(model_name_or_path)
    model.eval()
    return feature_extractor, model


@st.cache
def get_img_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=0.7, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()
    if id2label is not None:
        labels = [id2label[x] for x in labels]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    return fig2img(plt.gcf())


def make_prediction(img, feature_extractor, model):
    inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(**inputs)
    img_size = torch.tensor([tuple(reversed(img.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    return processed_outputs[0]


def main():
    st.write('''A.I Quality Control Assistive Model''' )
    option = st.selectbox("META (Facebook) DEtection TRansformer (DETR) Model", ("facebook/detr-resnet-50", "" ))
    feature_extractor, model = get_hf_components(option)
    url = st.text_input("Insert Image URL", "http://images.cocodataset.org/val2017/000000039769.jpg")
    img = get_img_from_url(url)
    processed_outputs = make_prediction(img, feature_extractor, model)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.7)
    viz_img = visualize_prediction(img, processed_outputs, threshold, model.config.id2label)
    st.image(viz_img)


if __name__ == "__main__":
    main()
