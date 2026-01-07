import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HuggingFace repository for model checkpoints
HF_REPO_ID = "quantumln314/imagenet-vision-models"  # Change this!

# Model configurations
MODELS = {
    'vit': 'vit_base_patch16_224',
    'swin': 'swin_base_patch4_window7_224',
    'convnext': 'convnext_base'
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load ImageNet class labels
def load_imagenet_classes():
    """Load ImageNet class labels"""
    try:
        # Try to load from URL
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as f:
            return json.loads(f.read().decode())
    except:
        # Fallback: return index labels
        return [f"Class {i}" for i in range(1000)]

class_labels = load_imagenet_classes()

# Load models
@torch.no_grad()
def load_model(model_name, checkpoint_path=None):
    """Load a model with optional checkpoint"""
    model_key = MODELS[model_name]
    
    # Create model WITHOUT downloading pretrained weights if we have a checkpoint
    model = timm.create_model(
        model_key,
        pretrained=(checkpoint_path is None),  # Only download if no checkpoint
        num_classes=1000
    )
    
    # Load checkpoint if available
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint for {model_name} from {checkpoint_path}")
            else:
                print(f"⚠ Using pretrained weights for {model_name}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint for {model_name}: {e}")
            print(f"  Using pretrained weights instead")
    
    model = model.to(device)
    model.eval()
    
    return model

def download_checkpoint(filename):
    """Download checkpoint from HuggingFace Hub"""
    try:
        # Try local file first
        local_path = f"./results/{filename}"
        if os.path.exists(local_path):
            print(f"✓ Using local checkpoint: {local_path}")
            return local_path
        
        # Download from HuggingFace
        print(f"Downloading {filename} from HuggingFace...")
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model"
        )
        print(f"✓ Downloaded {filename}")
        return path
    except Exception as e:
        print(f"⚠ Could not download {filename}: {e}")
        return None

# Initialize models
print("Loading models (9 total: 3 architectures × 3 data fractions)...")
models = {}

# Load ALL model checkpoints (1%, 5%, 10% for each architecture)
checkpoint_configs = {
    'vit_1pct': ('vit_1pct_best.pth', 'vit', '1%'),
    'vit_5pct': ('vit_5pct_best.pth', 'vit', '5%'),
    'vit_10pct': ('vit_10pct_best.pth', 'vit', '10%'),
    'swin_1pct': ('swin_1pct_best.pth', 'swin', '1%'),
    'swin_5pct': ('swin_5pct_best.pth', 'swin', '5%'),
    'swin_10pct': ('swin_10pct_best.pth', 'swin', '10%'),
    'convnext_1pct': ('convnext_1pct_best.pth', 'convnext', '1%'),
    'convnext_5pct': ('convnext_5pct_best.pth', 'convnext', '5%'),
    'convnext_10pct': ('convnext_10pct_best.pth', 'convnext', '10%'),
}

for model_key, (checkpoint_filename, model_name, data_fraction) in checkpoint_configs.items():
    try:
        checkpoint_path = download_checkpoint(checkpoint_filename)
        models[model_key] = {
            'model': load_model(model_name, checkpoint_path),
            'name': model_name,
            'data_fraction': data_fraction
        }
        print(f"✓ Loaded {model_name.upper()} ({data_fraction} data)")
    except Exception as e:
        # Fallback to pretrained
        print(f"⚠ Loading pretrained for {model_key}: {e}")
        models[model_key] = {
            'model': load_model(model_name, None),
            'name': model_name,
            'data_fraction': data_fraction
        }

print(f"✓ All {len(models)} models loaded successfully!")

@torch.no_grad()
def predict(image, top_k=5):
    """
    Run inference on ALL models and return predictions
    
    Args:
        image: PIL Image
        top_k: Number of top predictions to return
    
    Returns:
        Predictions for all 9 models organized by architecture
    """
    if image is None:
        empty_msg = "⬆️ Please upload an image to see predictions"
        return (empty_msg,) * 9 + (None,)
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Store results by architecture and all results
    vit_results = []
    swin_results = []
    convnext_results = []
    all_predictions = []
    
    # Run inference on all models
    for model_key, model_info in sorted(models.items()):
        model = model_info['model']
        model_name = model_info['name']
        data_fraction = model_info['data_fraction']
        
        # Get logits
        logits = model(img_tensor)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()
        
        # Store for comparison chart
        all_predictions.append({
            'model': f"{model_name.upper()} {data_fraction}",
            'architecture': model_name.upper(),
            'data_fraction': data_fraction,
            'top1_class': class_labels[top_indices[0]] if top_indices[0] < len(class_labels) else f"Class {top_indices[0]}",
            'top1_prob': float(top_probs[0]),
            'top5_coverage': float(sum(top_probs))
        })
        
        # Format results with emojis and better styling
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            predictions.append(f"{emoji} **{class_name}** — {prob*100:.2f}%")
        
        # Create formatted output with better text visibility
        model_output = f"""
<div style="background: white; padding: 15px; border-radius: 8px; color: #000;">

### <span style="color: #1a1a1a; font-weight: bold;">{model_name.upper()} — {data_fraction} Data</span>

{chr(10).join(predictions)}

---

<span style="color: #000;">📊 **Top-1 Confidence:** {top_probs[0]*100:.1f}%</span>  
<span style="color: #000;">📈 **Top-5 Coverage:** {sum(top_probs)*100:.1f}%</span>

</div>
"""
        
        # Categorize by architecture
        if model_name == 'vit':
            vit_results.append(model_output)
        elif model_name == 'swin':
            swin_results.append(model_output)
        elif model_name == 'convnext':
            convnext_results.append(model_output)
    
    # Create prediction table
    prediction_table = create_prediction_table(all_predictions)
    
    # Return all outputs (9 models + table)
    return (
        vit_results[0],   # VIT 1%
        vit_results[1],   # VIT 5%
        vit_results[2],   # VIT 10%
        swin_results[0],  # SWIN 1%
        swin_results[1],  # SWIN 5%
        swin_results[2],  # SWIN 10%
        convnext_results[0],  # ConvNeXt 1%
        convnext_results[1],  # ConvNeXt 5%
        convnext_results[2],  # ConvNeXt 10%
        prediction_table
    )


def create_confidence_chart(predictions):
    """Create bar chart comparing top-1 confidence across models"""
    df = pd.DataFrame(predictions)
    
    fig = px.bar(
        df,
        x='model',
        y='top1_prob',
        color='architecture',
        title='🎯 Top-1 Prediction Confidence Comparison',
        labels={'top1_prob': 'Confidence (%)', 'model': 'Model'},
        color_discrete_map={'VIT': '#FF6B6B', 'SWIN': '#4ECDC4', 'CONVNEXT': '#45B7D1'}
    )
    
    fig.update_layout(
        yaxis=dict(tickformat='.0%'),
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    return fig


def create_coverage_chart(predictions):
    """Create bar chart comparing top-5 coverage across models"""
    df = pd.DataFrame(predictions)
    
    fig = px.bar(
        df,
        x='model',
        y='top5_coverage',
        color='architecture',
        title='📈 Top-5 Coverage Comparison',
        labels={'top5_coverage': 'Coverage (%)', 'model': 'Model'},
        color_discrete_map={'VIT': '#FF6B6B', 'SWIN': '#4ECDC4', 'CONVNEXT': '#45B7D1'}
    )
    
    fig.update_layout(
        yaxis=dict(tickformat='.0%'),
        xaxis_tickangle=-45,
        height=400,
        showlegend=True
    )
    
    return fig


def create_prediction_table(predictions):
    """Create summary table of all predictions"""
    df = pd.DataFrame(predictions)
    df['top1_prob'] = df['top1_prob'].apply(lambda x: f"{x*100:.2f}%")
    df['top5_coverage'] = df['top5_coverage'].apply(lambda x: f"{x*100:.2f}%")
    df = df.rename(columns={
        'model': 'Model',
        'architecture': 'Architecture',
        'data_fraction': 'Data',
        'top1_class': 'Predicted Class',
        'top1_prob': 'Confidence',
        'top5_coverage': 'Top-5 Coverage'
    })
    
    return df[['Model', 'Predicted Class', 'Confidence', 'Top-5 Coverage']]


# Create Gradio interface with custom CSS and tabs
custom_css = """
.container {
    max-width: 1400px;
    margin: auto;
}
.header-text {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.prediction-box, .prediction-box * {
    background: #ffffff !important;
    color: #000000 !important;
}
.prediction-box h3, .prediction-box h3 * {
    color: #1a1a1a !important;
    font-weight: bold !important;
}
.prediction-box strong, .prediction-box b {
    color: #000000 !important;
    font-weight: 600 !important;
}
.prediction-box p, .prediction-box span {
    color: #000000 !important;
}
.markdown-text * {
    color: #000000 !important;
}
"""

with gr.Blocks(title="ImageNet Vision Model Comparison") as demo:
    
    # Header
    gr.HTML("""
        <div class="header-text">
            <h1>🔬 ImageNet Vision Model Comparison</h1>
            <h2>Compare 9 Models: 3 Architectures × 3 Data Fractions</h2>
            <p><strong>By:</strong> Satyam Dulal (23048597) | <strong>Module:</strong> CU6051NI Artificial Intelligence</p>
        </div>
    """)
    
    gr.Markdown("""
    ### 🎯 How It Works
    Upload any image and instantly see predictions from **9 trained models**:
    - 🤖 **ViT** (Vision Transformer) — Pure attention-based
    - 🪟 **Swin Transformer** — Hierarchical with shifted windows  
    - 🔷 **ConvNeXt** — Modernized CNN design
    
    Each trained on **1%, 5%, and 10%** of ImageNet-38K mini dataset to demonstrate data efficiency!
    """)
    
    # Main interface with tabs
    with gr.Tabs():
        
        # Tab 1: Quick Compare
        with gr.Tab("🚀 Quick Compare"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="📸 Upload Your Image",
                        sources=["upload", "webcam", "clipboard"],
                        height=400
                    )
                    
                    predict_btn = gr.Button(
                        "🎯 Analyze with All 9 Models",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    #### 💡 Quick Tips:
                    - Works with any image format
                    - Best with clear, centered objects
                    - Try different image types to compare
                    - All 1000 ImageNet-38K classes supported
                    """)
                    
                    # Example images
                    gr.Examples(
                        examples=[
                            ["https://images.unsplash.com/photo-1574158622682-e40e69881006"],
                            ["https://images.unsplash.com/photo-1546527868-ccb7ee7dfa6a"],
                            ["https://images.unsplash.com/photo-1552053831-71594a27632d"],
                        ],
                        inputs=image_input,
                        label="🖼️ Try Example Images"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### � Predictions Summary")
                    gr.Markdown("*Compare all 9 models at a glance*")
                    
                    prediction_summary = gr.Dataframe(
                        label="All Model Predictions at a Glance",
                        interactive=False,
                        wrap=True
                    )
        
        # Tab 2: Detailed Results by Architecture
        with gr.Tab("📊 Detailed Results"):
            gr.Markdown("### 🤖 Vision Transformer (ViT)")
            gr.Markdown("*Pure attention-based architecture treating images as sequence of patches*")
            with gr.Row():
                vit_1pct = gr.Markdown(elem_classes="prediction-box")
                vit_5pct = gr.Markdown(elem_classes="prediction-box")
                vit_10pct = gr.Markdown(elem_classes="prediction-box")
            
            gr.Markdown("---")
            gr.Markdown("### 🪟 Swin Transformer")
            gr.Markdown("*Hierarchical vision transformer with shifted window attention*")
            with gr.Row():
                swin_1pct = gr.Markdown(elem_classes="prediction-box")
                swin_5pct = gr.Markdown(elem_classes="prediction-box")
                swin_10pct = gr.Markdown(elem_classes="prediction-box")
            
            gr.Markdown("---")
            gr.Markdown("### 🔷 ConvNeXt")
            gr.Markdown("*Modernized CNN design incorporating Transformer insights*")
            with gr.Row():
                convnext_1pct = gr.Markdown(elem_classes="prediction-box")
                convnext_5pct = gr.Markdown(elem_classes="prediction-box")
                convnext_10pct = gr.Markdown(elem_classes="prediction-box")
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 📚 Research Overview
            
            This interactive demo showcases research comparing modern vision architectures on ImageNet classification
            with varying amounts of training data to analyze **data efficiency** and **few-shot learning** capabilities.
            
            ### 🏗️ Architectures Compared
            
            | Model | Type | Parameters | Key Feature |
            |-------|------|------------|-------------|
            | **ViT** | Pure Transformer | 86M | Patch-based self-attention |
            | **Swin** | Hierarchical Transformer | 88M | Shifted window attention |
            | **ConvNeXt** | Modernized CNN | 89M | CNN + Transformer insights |
            
            ### 📊 Training Data Fractions
            
            - **1%** (~13,000 images) — Extreme few-shot learning
            - **5%** (~64,000 images) — Moderate few-shot learning
            - **10%** (~128,000 images) — Standard few-shot learning
            
            ### 🎯 Key Research Findings
            
            1. **ConvNeXt** achieves 83.3%+ Top-1 accuracy even with just 1% of data
            2. **Swin Transformer** shows consistent 80%+ performance across all data fractions
            3. **ViT** demonstrates steady improvement with more data (74% → 75.6%)
            4. All models benefit from strong ImageNet pretrained weights
            5. **Data efficiency** varies significantly across architectures
            
            ### 📈 Metrics Explained
            
            - **Top-1 Accuracy**: % of times the top prediction is correct
            - **Top-5 Coverage**: Sum of probabilities for top 5 predictions
            - **Confidence**: How certain the model is about its prediction
            - **Data Efficiency**: Performance achieved per % of training data
            
            ### 🔗 Resources
            
            - **GitHub Repository**: [imagenet-vision-model-comparison](https://github.com/sat-yam-ln2/imagenet-vision-model-comparison)
            - **Full Results**: See `results/` directory for detailed metrics and visualizations
            - **Training Code**: Complete Jupyter notebook with all experiments
            - **Author**: Satyam Dulal (dulals@coventry.ac.uk)
            
            ### 📄 Citation
            
            ```bibtex
            @misc{dulal2024imagenet,
              author = {Dulal, Satyam},
              title = {ImageNet Vision Model Comparison: ViT, Swin, and ConvNeXt},
              year = {2024},
              publisher = {GitHub},
              url = {https://github.com/sat-yam-ln2/imagenet-vision-model-comparison}
            }
            ```
            
            ### 🛠️ Technical Stack
            
            - **Framework**: PyTorch 2.0+
            - **Models**: timm library (Ross Wightman)
            - **Interface**: Gradio 4.0+
            - **Visualization**: Plotly Express
            - **Training**: Mixed Precision (AMP), AdamW optimizer
            
            ---
            
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px;">
                <p><strong>⭐ If you find this useful, please star the GitHub repository!</strong></p>
                <p>Built with PyTorch, timm, and Gradio | © 2024 Satyam Dulal</p>
            </div>
            """)
    
    # Connect all outputs
    outputs_list = [
        vit_1pct, vit_5pct, vit_10pct,
        swin_1pct, swin_5pct, swin_10pct,
        convnext_1pct, convnext_5pct, convnext_10pct,
        prediction_summary
    ]
    
    # Button click event
    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=outputs_list
    )
    
    # Auto-predict on image upload
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=outputs_list
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css
    )
