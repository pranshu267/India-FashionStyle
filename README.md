# India-FashionStyle

The **Indo Fashion CLIP Model** project is a multimodal application that leverages a fine-tuned CLIP model to find and display the top 3 matching Indian fashion images based on a text description. This project demonstrates the integration of AI with Indian fashion, providing a unique experience for users to explore various styles and clothing.

## Features

- **Multimodal Search**: Uses both text and image data for searching.
- **Fine-Tuned Model**: Utilizes a CLIP model fine-tuned on an Indian fashion dataset.
- **Fast Retrieval**: Employs FAISS, a high-performance vector search library, for efficient similarity search.
- **Streamlit Interface**: Provides a user-friendly web interface for easy interaction.

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional but recommended for performance)
- [Streamlit](https://streamlit.io/)
- Required Python libraries: `torch`, `transformers`, `PIL`, `faiss-cpu` or `faiss-gpu`

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/indo-fashion-clip-model.git
   cd indo-fashion-clip-model

2. **Download the required libraries:**

3. **Run the Jupyter Notebook:**

    Utilize the Jupyter Notebook to conduct hyperparameter tuning and execute the code for fine-tuning the base CLIP model. Ensure to save the fine-tuned model for running the app.

4. **Create FAISS Index for the Image Embeddings:**

   Run the `create_db.py` script located in the `scripts` folder to generate embeddings for the images and store them in the FAISS index.

    ```bash
   python create_dp.py

5. **Run the Streamlit App:**

   Execute the `app.py` file located in the `source` folder to launch the Streamlit application and begin exploring Indo-inspired clothing!

    ```bash
   python app.py