<h2>2018 Lane Detection Project</h2>

<p>
Originally developed in 2018 during my sophomore year, this project
focused on camera-based lane detection for a cruise control system.
The original implementation primarily detected straight lane markings
and had limitations in handling curved roads.
</p>

<h3>Original Lane Detection Algorithm</h3>

<ol>
  <li><strong>Preprocessing:</strong> Converted the input image to grayscale and applied Gaussian blur to reduce noise.</li>
  <li><strong>Edge Detection:</strong> Used Canny edge detection to extract lane markings.</li>
  <li><strong>Region of Interest (ROI):</strong> Restricted detection to the road area to exclude irrelevant edges.</li>
  <li><strong>Line Detection:</strong> Applied Probabilistic Hough Line Transform to detect line segments.</li>
  <li><strong>Lane Classification:</strong> Filtered line segments by slope angle and classified left and right lane candidates based on slope signs.</li>
  <li><strong>Intersection Calculation:</strong> Calculated the intersection point of the representative left and right lane lines. Extracted the x-coordinate of the intersection point and used it as the steering input for the RC car.</li>
</ol>

<p align="center">
  <img width="642" height="578" alt="2018 Lane Detection Result" src="https://github.com/user-attachments/assets/4b529693-6aca-4097-a344-0115bd66395c" />
</p>

<h3>Lane following test with Prototype</h3>
<p align="center">
  <img height="300" alt="20181028_093728_1(1)" src="https://github.com/user-attachments/assets/5b4345da-7455-4df4-b900-dc1c5b09f584" />
  <img height="300" alt="20181030_012443(1)" src="https://github.com/user-attachments/assets/742aaf59-a401-42ef-aade-fe801de7ba87" />
  <img height="300" alt="20181107_145457(1)" src="https://github.com/user-attachments/assets/3771a184-ba83-484f-bb20-41c6b163350e" />
  <img height="300" alt="VAP KakaoTalk_Video_20181106_1804_25_451 1" src="https://github.com/user-attachments/assets/0c7dcdc6-90dd-4088-9431-753b01a8bac2" />

</p>

<h2>2026 Lane Detection Project</h2>

<p>
I revisited the project to review its original implementation,
identify its limitations, and improve the lane detection algorithm.
The updated version was reimplemented in C++ using OpenCV,
with additional techniques to detect and track curved lane boundaries.
</p>

<h3>Improved Lane Detection Algorithm</h3>

<ol>
  <li><strong>Bird's-Eye View (BEV) Transformation:</strong> Applied perspective transformation to convert the road image into a top-down view, making lane boundaries easier to analyze.</li>
  <li><strong>Region of Interest (ROI):</strong> Restricted the detection area to the central region of the BEV image to reduce irrelevant information.</li>
  <li><strong>Lane Feature Extraction:</strong> Converted the image to grayscale and combined brightness thresholding with Sobel X edge detection to identify potential lane markings. Applied morphological operations to reduce noise and refine the binary mask.</li>
  <li><strong>Histogram Analysis:</strong> Analyzed the lower portion of the binary image using a horizontal histogram to estimate the initial positions of the left and right lane boundaries.</li>
  <li><strong>Sliding Window Search:</strong> Divided the image into multiple horizontal windows and collected lane pixels around the estimated lane positions. Used previous-frame polynomial estimates to guide the search when available.</li>
  <li><strong>Polynomial Fitting:</strong> Fitted a second-degree polynomial to the collected lane pixels to represent curved lane boundaries.</li>
  <li><strong>Temporal Smoothing:</strong> Applied Exponential Moving Average (EMA) to the detected lane curves to reduce frame-to-frame fluctuations and improve tracking stability.</li>
  <li><strong>Lane Visualization and Offset Calculation:</strong> Drew the detected lane boundaries, transformed the result back onto the original camera image, and calculated the horizontal offset between the estimated lane center and the image center.</li>
</ol>

<p align="center">
  <img width="642" height="518" alt="2026 Lane Detection Result" src="https://github.com/user-attachments/assets/9fd00d9b-ad94-45f6-89c1-ee3a5c490cd9" />
</p>
