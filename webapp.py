"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import datetime
import cv2
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        before_time = datetime.datetime.now()
        results = model([img])
        elapsed_time = datetime.datetime.now() - before_time
        
        print(f"Elapsed time: {elapsed_time}")
        print(f"results: {results.pandas().xyxy[0]}")

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        
        if not results.pandas().xyxy[0].empty:
        
            import PIL as pil
            height, width = img.size
            image = Image.new("RGB", (height, width), (255, 255, 255))
            
            fontsize = 50
            font = pil.ImageFont.truetype("arial.ttf", fontsize)
            draw = pil.ImageDraw.Draw(image)
            draw.text((-1,-1), f"Inference Time: {elapsed_time} \n Confidence: {results.pandas().xyxy[0].confidence[0]} \n Object: {results.pandas().xyxy[0].name[0]}", (0,0,0), font=font)
            
            img = pil.Image.open(img_savename)
            print(f"ext_size: {img.size}")
            print(f"blank_size: {image.size}")
            
            get_concat_h(img, image).save('static/pillow_concat_h.jpg')
    
            # show the output image
            #cv2.imshow('man_image.jpeg', im_h)
            
            #im_h.save("static/a_test.png")
            
            return redirect('static/pillow_concat_h.jpg')
        
        else:
            return redirect(img_savename)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5','custom', path='best.pt',force_reload=True)
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
