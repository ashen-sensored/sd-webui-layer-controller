The script allows you to control the layers of the UNet model by changing the weights of each layer. The weights are changed by using sliders. The sliders are located in the Layer Controller section of the UI. There are 25 sliders, one for each layer of the UNet model. Each slider has a range from -1 to 2, with 1 being the default value. The values can be changed by clicking and dragging on the slider or by typing in a value into the text box next to it.

The script also allows you to analyze how each layer contributes to an image by selecting Enable Layer Analysis and clicking on Analyze Layer Contribution button after image Generation finished. This will display an image for each layer that shows how much it contributed to an image.

Added layer shadowing option for better single layer isolation.
