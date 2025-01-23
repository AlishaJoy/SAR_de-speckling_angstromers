# S1 images : deburst 
1. Deburst the Image
What is Deburst?
Deburst removes overlapping seams between bursts in Sentinel-1 SLC (Single Look Complex) data. Sentinel-1 SLC products are stored in bursts, and to process the entire scene as one continuous image, debursting is required.

Detailed Steps:
Open Your Data:

Launch SNAP.
Open the .SAFE file in your S1A-1W-SLC folder by navigating to File → Open Product and selecting manifest.safe.
Visualize Bands:

In the Product Explorer window, expand the Bands folder. You should see multiple bands, such as i_IW1_VH, q_IW1_VH, etc.
Perform Deburst:

In the menu bar, go to Radar → Sentinel-1 TOPS → TOPSAR-Deburst.
A dialog box will appear.
Input Product: Select your Sentinel-1 product from the dropdown.
Sub-swath: Choose the relevant sub-swath (e.g., IW1, IW2, or IW3).
Polarization: Select the polarization type (e.g., VH or VV).
Click Run.
The result will be a deburst product, which appears in the Product Explorer window as a new entry.
# to get intensity 
Intensity Calculation:
Raster → Band Maths
New Band Parameters:
Name: Intensity_IW1_VH
Type: float32
Expression: sqrt(i_IW1_VH^2 + q_IW1_VH^2)
Confirm calculation
