# Image_Resizing_With_Seam_Carving

This project was an attempt to use seam carving to resize images. Under this, I have gone through the details of seam carving and successfully resized some images. I used dynamic programming concept, python programming language & jupyter Notebook software for the implementation.


# Seam Carving
Seam carving is a way to crop images without losing important content in the image.

The algorithm works as follows:
1. Assign an energy value to every pixel
2. Find an 8-connected path of the pixels with the least energy
3. Delete all the pixels in the path
4. Repeat 1-3 till the desired number of rows/columns are deleted

Reference: https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
