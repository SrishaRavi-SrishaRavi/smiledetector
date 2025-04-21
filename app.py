{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "001f1f02-266c-4264-8d0d-df0fbb36d842",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7861\n",
      "\n",
      "To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7861/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import gradio as gr\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "from joblib import load\n",
    "\n",
    "\n",
    "model = load(\"smile_detector_model.joblib\")\n",
    "scaler = load(\"scaler.joblib\")\n",
    "\n",
    "\n",
    "def smile_detector(img):\n",
    "    img = img.convert(\"L\").resize((64, 64))           # Convert to grayscale and resize\n",
    "    img_array = np.array(img).flatten().reshape(1, -1)  # Flatten and reshape for model\n",
    "    img_scaled = scaler.transform(img_array)          # Scale it\n",
    "    prediction = model.predict(img_scaled)[0]\n",
    "    return \"😁 Smile Detected!\" if prediction == 1 else \"😐 No Smile Detected\"\n",
    "\n",
    "# Gradio interface\n",
    "demo = gr.Interface(\n",
    "    fn=smile_detector,\n",
    "    inputs=gr.Image(type=\"pil\"),\n",
    "    outputs=\"text\",\n",
    "    title=\"Smile Detector 😊\",\n",
    "    description = \"Hello peeps!! I have a created an AI model,photo, please give it a try!\",\n",
    "    theme = \"default\"\n",
    ")\n",
    "\n",
    "demo.launch()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58331496-6d75-46a3-a988-e8d9132480a5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
