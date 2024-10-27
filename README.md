# Skin Disease Classifier

This is a Streamlit application that classifies images of skin diseases using a pre-trained ResNet-50 model.

## Requirements

To run this application, you need to install the required packages. You can do this by creating a `requirements.txt` file and installing the dependencies with pip.

### Install Python 3

1. Download Python 3 from the official website: [python.org](https://www.python.org/downloads/).
2. Follow the installation instructions for your operating system.
3. Ensure to check the box that says **"Add Python to PATH"** during installation.

### Set Up a Virtual Environment

1. Open a terminal or command prompt.
2. Navigate to your project directory:

   ```bash
   cd path/to/your/project
   ```

3. Create a virtual environment:

   ```bash
   python -m venv env
   ```

4. Activate the virtual environment:

   - On Windows:

     ```bash
     .\env\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source env/bin/activate
     ```

### Install Requirements

1. Create a `requirements.txt` file with the following content:

   ```plaintext
   streamlit
   torch
   torchvision
   Pillow
   numpy
   ```

2. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

Once the dependencies are installed, you can run the Streamlit application:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your main Python script if it's different.

## How to Use

1. After starting the application, a new tab will open in your default web browser.
2. Upload an image of skin disease (jpg, jpeg, png).
3. The model will classify the uploaded image and display the predicted class.

## Model Details

The application uses a pre-trained ResNet-50 model fine-tuned on a dataset of skin diseases. Ensure that you have the trained model file (`entire_model.pth`) in the same directory as the application script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
