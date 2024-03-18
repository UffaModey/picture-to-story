from flask import Flask
import json
import os
from dotenv import load_dotenv

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
app = Flask(__name__)
# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.getenv("VISION_ENDPOINT")
    key = os.getenv("VISION_KEY")
    # Set your OpenAI API key here

except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
cl = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


@app.route('/')
def tell_story_from_pictures():
    # Replace the below URL with the one you wish to analyze
    with open("static/images/cecelia-chang-JxaTUmfmBGM-unsplash.jpg", "rb") as f:
        image_data_one = f.read()
    with open("static/images/richard-stachmann-3g9E6n15e7E-unsplash.jpg", "rb") as f:
        image_data_two = f.read()
    with open("static/images/tommao-wang-actWFB5jklQ-unsplash.jpg", "rb") as f:
        image_data_three = f.read()

    result_one = cl.analyze(
        image_data=image_data_one,
        visual_features=["CAPTION", "READ"],
        gender_neutral_caption=True,
    )
    result_two = cl.analyze(
        image_data=image_data_two,
        visual_features=["CAPTION", "READ"],
        gender_neutral_caption=True,
    )
    result_three = cl.analyze(
        image_data=image_data_three,
        visual_features=["CAPTION", "READ"],
        gender_neutral_caption=True,
    )
    analysis_result = json.dumps({
        "caption_one": result_one.caption.text,
        "caption_confidence_one": result_one.caption.confidence,
        "caption_two": result_two.caption.text,
        "caption_confidence_two": result_two.caption.confidence,
        "caption_three": result_three.caption.text,
        "caption_confidence_three": result_three.caption.confidence,
    })
    data = json.loads(analysis_result)

    prompt = f"Write a short story of 50 words that includes these elements: {data['caption_one']}, \
            {data['caption_two']}, and {data['caption_three']}."

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a children's book author skilled in adventure and fiction novels set in 1980."},
                {"role": "user", "content": prompt}
            ]
        )


    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
    result = completion.choices[0].message
    return {
        "result": result.content
    }


if __name__ == '__main__':
    app.run(debug=True)
