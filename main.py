import cv2
import pathlib
import pyautogui
from tkinter import filedialog, Menu
import customtkinter as ctk
from PIL import Image, ImageTk
import threading

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image

import cv2
import numpy as np



class SketchImage:
    def __init__(self, root):
        self.window = root
        self.window.geometry("1000x750")
        self.window.title('Sketch Creator')
        self.window.resizable(False, False)

        self.width = 740
        self.height = 480

        self.Image_Path = ''
        self.SketchImg = None
        self.original_image = None

        # Set the theme
        ctk.set_appearance_mode("Dark")  # Set Dark Mode
        ctk.set_default_color_theme("green")  # Set a green theme

        # Menu Bar
        self.menubar = Menu(self.window)
        self.window.config(menu=self.menubar)

        # File Menu
        self.file_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Image", command=self.Open_Image)
        self.file_menu.add_command(label="Exit", command=self.Exit)

        # Sketch Menu
        self.sketch_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Sketch", menu=self.sketch_menu)
        self.sketch_menu.add_command(label="Create Pencil Sketch", command=self.CreateSketch)
        self.sketch_menu.add_command(label="Apply Artistic Filter", command=self.ApplyArtisticFilter)
        self.sketch_menu.add_command(label="Apply Van Gogh Style", command=self.Apply_Van_Gogh_Style)


        # Save Menu
        self.save_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Image", command=self.Save_Image)

        # Create UI Elements

        # Image display frame
        self.frame = ctk.CTkFrame(self.window, width=self.width, height=self.height)
        self.frame.place(x=130, y=20)

        # Intensity Control
        self.intensity_label = ctk.CTkLabel(self.window, text="Intensity")
        self.intensity_label.place(x=300, y=520)

        self.intensity = ctk.CTkSlider(self.window, from_=5, to=155, command=self.Update_Preview)
        self.intensity.set(37)
        self.intensity.place(x=300, y=550)

        # Status Label
        self.status_label = ctk.CTkLabel(self.window, text="Status: No image loaded.", text_color="red")
        self.status_label.place(x=20, y=600)

        # Loading Spinner (Hidden by default, using label for animation)
        self.loading_spinner_label = ctk.CTkLabel(self.window, text="Loading...", text_color="white", font=("Arial", 16))
        self.loading_spinner_label.place(x=850, y=500)
        self.loading_spinner_label.grid_forget()

    def Update_Status(self, message, color="red"):
        self.status_label.configure(text=f"Status: {message}", text_color=color)

    def Open_Image(self):
        self.Clear_Screen()
        self.Image_Path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"),))
        if self.Image_Path:
            self.original_image = cv2.imread(self.Image_Path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert for PIL
            self.Show_Image(self.Image_Path)
            self.Update_Status("Image loaded successfully.", "green")

    def Show_Image(self, Img):
        image = Image.open(Img)
        resized_image = image.resize((self.width, self.height))
        self.img = ImageTk.PhotoImage(resized_image)

        label = ctk.CTkLabel(self.frame, image=self.img)
        label.pack()

    def CreateSketch(self):
        if not self.Image_Path:
            pyautogui.alert("Please load an image first!")
            return

        # Show loading spinner
        self.loading_spinner_label.grid(row=1, column=1)
        self.Update_Status("Creating sketch...", "blue")

        # Start the sketch creation in a separate thread
        threading.Thread(target=self.Generate_Sketch).start()

    def Generate_Sketch(self):
        Img = cv2.imread(self.Image_Path)
        Img = cv2.resize(Img, (740, 480))

        # Create pencil sketch
        GrayImg = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        InvertImg = cv2.bitwise_not(GrayImg)
        SmoothImg = cv2.medianBlur(InvertImg, self.intensity.get())
        IvtSmoothImg = cv2.bitwise_not(SmoothImg)
        self.SketchImg = cv2.divide(GrayImg, IvtSmoothImg, scale=250)

        # Update UI with the sketch
        self.Show_Sketch(self.SketchImg)
        self.Update_Status("Pencil sketch created.", "green")

        # Hide the loading spinner
        self.loading_spinner_label.grid_forget()

    def Show_Sketch(self, Sketch):
    # Remove all widgets in the frame before showing the new image
        for widget in self.frame.winfo_children():
            widget.destroy()

        final_image = Image.fromarray(cv2.cvtColor(Sketch, cv2.COLOR_BGR2RGB))
        resized_image = final_image.resize((self.width, self.height))
        self.img = ImageTk.PhotoImage(resized_image)

        label = ctk.CTkLabel(self.frame, image=self.img)
        label.pack()


    def ApplyArtisticFilter(self):
        if len(self.Image_Path) == 0:
            pyautogui.alert("Please load an image first!")
            return
        try:
            Img = cv2.imread(self.Image_Path)
            Img = cv2.resize(Img, (740, 480))

            # Apply bilateral filter (to simulate the brushstroke effect)
            img_filtered = cv2.bilateralFilter(Img, 5, 75, 75)

            # Convert the image to grayscale for edge detection
            gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

            # Apply median blur to smooth the image
            gray_blurred = cv2.medianBlur(gray, 5)

            # Create an edge mask by detecting edges
            edges = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 9)

            # Combine the filtered image with edges
            oil_painting_effect = cv2.bitwise_and(img_filtered, img_filtered, mask=edges)

            # Show the result
            self.Show_Sketch(oil_painting_effect)

        except Exception as e:
            error_message = f"An error occurred while applying the artistic filter: {str(e)}"
            print(error_message)  # Log the error in the console for more details
            pyautogui.alert(error_message)  # Show the error in an alert box

    def Apply_Van_Gogh_Style(self):
        if not self.Image_Path:
            pyautogui.alert("Please load an image first!")
            return

        # Load the content image (the one that will be transformed)
        content_image = self.Load_Image(self.Image_Path)

        # Load the style image (Van Gogh's "The Starry Night")
        style_path = 'C:\\Sashreek\\Pyhton\\img2skch\\91q9JFqoSaS._AC_UF894,1000_QL80_.jpg'  # You need to specify the path to the Van Gogh image
        style_image = self.Load_Image(style_path)

        # Load the TensorFlow Hub model
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        # Perform style transfer
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

        # Convert the tensor to an image and display it
        self.Display_Stylized_Image(stylized_image)
        
    def Load_Image(self, path):
        # Function to load and process the image for NST
        max_dim = 512
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def Display_Stylized_Image(self, tensor):
        # Convert the tensor back to a displayable image
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        
        final_image = PIL.Image.fromarray(tensor)
        resized_image = final_image.resize((self.width, self.height))
        self.img = ImageTk.PhotoImage(resized_image)

        # Display the image on the screen
        for widget in self.frame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(self.frame, image=self.img)
        label.pack()


    def Save_Image(self):
        if self.SketchImg is None:
            pyautogui.alert("No sketch to save!")
            return

        filename = pyautogui.prompt("Enter the filename to save:")
        if filename:
            filename = filename + pathlib.Path(self.Image_Path).suffix
            cv2.imwrite(filename, self.SketchImg)
            pyautogui.alert(f"Image saved as {filename}")
            self.Update_Status(f"Image saved as {filename}.", "green")

    def Update_Preview(self, event=None):
        """Update the preview with current intensity adjustment."""
        if self.original_image is not None:
            Img = self.original_image
            GrayImg = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            InvertImg = cv2.bitwise_not(GrayImg)
            
            # Get intensity as an integer and ensure it's odd
            intensity_value = int(self.intensity.get())
            if intensity_value % 2 == 0:  # Ensure the value is odd
                intensity_value += 1
            
            # Apply median blur with the adjusted intensity value
            SmoothImg = cv2.medianBlur(InvertImg, intensity_value)
            IvtSmoothImg = cv2.bitwise_not(SmoothImg)
            preview_img = cv2.divide(GrayImg, IvtSmoothImg, scale=250)

            self.Show_Sketch(preview_img)


    def Clear_Screen(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def Exit(self):
        self.window.quit()

if __name__ == "__main__":
    root = ctk.CTk()
    app = SketchImage(root)
    root.mainloop()
