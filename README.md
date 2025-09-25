# Cultural-Heritage-Story-Teller
🌟 Overview
An intelligent AI system that "sees" cultural artifacts and weaves authentic stories about India's rich heritage. This project fine-tunes a Qwen2-VL 2B parameter vision-language model using a groundbreaking 3-stage progressive learning approach on 1,360 diverse regional images spanning India's vast cultural tapestry.

✨ What Makes This Special?
🎯 Revolutionary 3-Stage Training: Recognition → Understanding → Storytelling

🗺️ Pan-Indian Coverage: 5 geographical regions (North, South, East, West, Northeast)

🚀 Advanced LoRA Fine-tuning: 85% performance improvement achieved

📖 Authentic Narratives: Culturally accurate stories preserving heritage

💡 Research Innovation: Novel progressive learning methodology

🏛️ Cultural Impact
"Transforming silent cultural images into vivid, contextual narratives for digital heritage preservation"

This project addresses the critical need for digital cultural preservation by:

📚 Documenting Stories: Converting visual heritage into textual narratives

🌍 Accessibility: Making cultural knowledge accessible globally

🔬 Research Tool: Supporting cultural studies and heritage research

🎓 Education: Enhancing cultural education through AI storytelling

📊 Project Highlights
Metric	Achievement
Dataset Size	1,360 curated cultural images
Model Parameters	2B (Qwen2-VL) with LoRA fine-tuning
Performance Improvement	85% loss reduction (130 → 19.7)
Regional Coverage	5 major Indian geographical regions
Training Methodology	3-stage progressive learning
Cultural Accuracy	High-quality, authentic narratives
🏗️ 3-Stage Progressive Architecture
text
🖼️ Cultural Image Input
            ↓
    👁️ Qwen2-VL Vision Encoder
            ↓
    🧠 Language Model Processing
            ↓
┌─────────────────────────────────┐
│  Stage 1: Element Recognition  │ ← Identifies cultural elements
├─────────────────────────────────┤
│ Stage 2: Pattern Understanding │ ← Understands regional context
├─────────────────────────────────┤
│ Stage 3: Story Generation      │ ← Weaves authentic narratives
└─────────────────────────────────┘
            ↓
    📖 Cultural Story Output
Progressive Learning Stages
🔍 Cultural Element Recognition

Identifies temples, sculptures, architectural features

Recognizes traditional clothing, artifacts, decorations

Training: 1 epoch, LR: 2e-4

🎨 Cultural Pattern Understanding

Understands architectural styles (Dravidian, Mughal, etc.)

Analyzes regional patterns and significance

Training: 1 epoch, LR: 1e-4

📚 Cultural Story Generation

Generates authentic, contextual narratives

Preserves cultural accuracy and historical context

Training: 2 epochs, LR: 5e-5

🚀 Quick Start
Installation
bash
# Clone the repository
git clone https://github.com/PranavSingla122/Cultural-Heritage-Story-Teller.git
cd Cultural-Heritage-Story-Teller

# Install dependencies
pip install -r requirements.txt

# Install additional requirements for vision-language models
pip install transformers[vision] torch torchvision peft accelerate bitsandbytes

Example Output:

text
This magnificent Dravidian temple showcases the architectural brilliance of Tamil Nadu, 
with its towering gopuram adorned with intricate sculptures depicting celestial beings 
and divine stories. The stone carvings, weathered by centuries of devotion, tell tales 
of ancient rituals and spiritual practices that continue to resonate in South Indian 
culture today...
📁 Repository Structure
text
Cultural-Heritage-Story-Teller/
├── 🐍 train_progressive.py          # Main 3-stage training pipeline
├── 🔧 continue_training.py          # Optional optimization script
├── 🔮 inference.py                  # Story generation from trained model
├── 📊 dataset_utils.py              # Dataset loading and preprocessing
├── 🎛️ config.py                    # Training configuration
├── 📋 requirements.txt              # Python dependencies
├── 📁 data/                         # Dataset directory
│   └── cultural_dataset.json       # Your cultural image dataset
├── 📁 models/                       # Trained model checkpoints
│   └── cultural_vlm_trained/        # Final trained model
├── 📁 examples/                     # Usage examples and demos
└── 📖 README.md                     # This file
🎯 Training Your Own Model
Dataset Format
Your cultural dataset should follow this JSON structure:

json
{
  "image_path": "regional_images/South_India/temple_001.jpg",
  "region": "South_India",
  "real_features": {
    "detected_elements": ["temple", "gopuram", "sculpture"],
    "architecture": {"style": "Dravidian"},
    "colors": {"dominant": ["brown", "red"]},
    "scene": {"type": "religious"}
  },
  "unique_story": "This ancient Dravidian temple stands as a testament...",
  "cultural_context": "South Indian temple architecture represents..."
}
3-Stage Progressive Training
bash
# Run complete 3-stage progressive training
python train_progressive.py

# For testing with limited samples
python train_progressive.py --test_samples 100

# Single stage training (skip progressive approach)
python train_progressive.py --single_stage
Advanced Training Options
python
# Custom configuration
config = CulturalVLMConfig()
config.learning_rate = 2e-4
config.batch_size = 1
config.gradient_accumulation_steps = 16
config.lora_r = 8
config.lora_alpha = 32

# Enable/disable progressive training
config.enable_progressive = True
config.stage1_epochs = 1
config.stage2_epochs = 1  
config.stage3_epochs = 2
📈 Training Results
Progressive Training Performance
Stage	Task	Loss Reduction	Key Learning
1	Element Recognition	504 → 217	Cultural artifact identification
2	Pattern Understanding	217 → 201	Regional style comprehension
3	Story Generation	184 → 19.7	Authentic narrative creation
Regional Dataset Distribution
🏔️ North India: 160 samples (Rajasthan, Delhi, Himalayas)

🏛️ South India: 400 samples (Tamil Nadu, Karnataka, Kerala)

🎭 East India: 400 samples (West Bengal, Odisha, Jharkhand)

🏜️ West India: 200 samples (Gujarat, Maharashtra, Rajasthan)

🌄 Northeast India: 200 samples (Assam, Meghalaya, Sikkim)

🛠️ Technical Specifications
Model Architecture
Base Model: Qwen2-VL-2B-Instruct

Fine-tuning: LoRA (r=8, α=32, 7 target modules)

Quantization: 4-bit with bfloat16 precision

Context Length: 512 tokens

Image Resolution: 336×336 pixels

Training Configuration
Optimizer: AdamW with cosine scheduling

Memory Optimization: Gradient checkpointing + quantization

Hardware Requirements: 8GB+ GPU recommended

Training Time: ~3-5 hours for full pipeline
🤝 Contributing
We welcome contributions from cultural enthusiasts, AI researchers, and heritage preservationists!

How to Contribute
🍴 Fork the repository

🌿 Create a feature branch

Contribution Areas
🎨 Dataset Expansion: Adding more regional cultural images

🔬 Model Improvements: Enhancing training methodologies

🌍 Multilingual Support: Adding regional language stories

📱 Applications: Building demo apps and interfaces

📖 Documentation: Improving guides and examples

📜 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Usage Rights
✅ Commercial Use: Permitted

✅ Modification: Permitted

✅ Distribution: Permitted

✅ Patent Use: Permitted

⚠️ Trademark Use: Not permitted

⭐ Star this repository if it helped preserve cultural heritage through AI!
"Building bridges between ancient wisdom and modern technology"
