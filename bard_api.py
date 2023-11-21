import os
import time

# !pip install bardapi
from bardapi import Bard


# verify response
def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        # print("Response Error")
        return False
    return True


# build bard class
class Bard_Model():
    def __init__(self, token, patience=1, sleep_time=1):
        print("!!Bard Token:", token)
        self.patience = patience
        self.sleep_time = sleep_time
        self.model = Bard(token=token)

    def get_response(self, image_path, input_text):
        patience = self.patience
        while patience > 0:
            patience -= 1
            # print(f"Patience: {patience}")
            try:
                # final_image_path = os.path.join("/local1/rwadhawan/document_understanding/datasets/training/mplug_owl/test", image_path)
                # assert os.path.exists(final_image_path)
                
                image = open(image_path, 'rb').read() # (jpeg, png, webp) are supported.
                print(image_path)
                response = self.model.ask_about_image(input_text, image)
                response = response['content'].strip()
                if verify_response(response):
                    # print("# Verified Response")
                    # print(response)
                    return response
                else:
                    print(response)
            except Exception as e:
                print(e)
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
        return ""