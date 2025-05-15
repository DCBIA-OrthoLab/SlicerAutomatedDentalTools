import qt


def createButton(name, callback=None, isCheckable=False, icon=None, toolTip="", parent=None):
    """Helper function to create a button with a text, callback on click and checkable status

    :param name: Text of the button
    :param callback: Callback called on click if not None
    :param isCheckable: When True, button can be checked (click will send check state)
    :param icon: QIcon to use for button
    :param toolTip: Tooltip displayed when hovering button
    :param qtProperty: Optional list of property name and property value to set to button
    :param parent: QWidget parent of this button

    :returns: QPushButton
    """
    button = qt.QPushButton(name, parent)
    if callback is not None:
        button.connect("clicked(bool)", callback)
    if icon:
        button.setIcon(icon)
    button.setCheckable(isCheckable)
    button.setToolTip(toolTip)
    return button


def addInCollapsibleLayout(childWidget, parentLayout, collapsibleText, isCollapsed=True):
    """
    Wraps input childWidget into a collapsible button attached to input parentLayout.
    collapsibleText is writen next to collapsible button. Initial collapsed status is customizable
    (collapsed by default)
    """
    import ctk
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = collapsibleText
    collapsibleButton.collapsed = isCollapsed
    parentLayout.addWidget(collapsibleButton)
    collapsibleButtonLayout = qt.QVBoxLayout()
    collapsibleButtonLayout.addWidget(childWidget)
    collapsibleButton.setLayout(collapsibleButtonLayout)


def set3DViewBackgroundColors(topColor, bottomColor):
    """ Set the background color as a gradient between the top and bottom colors

    :param topColor: (r, g, b) floats between 0 and 1
    :param bottomColor: (r, g, b) floats between 0 and 1
    """
    import slicer
    viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
    viewNode.SetBackgroundColor(bottomColor)
    viewNode.SetBackgroundColor2(topColor)


def setBoxAndTextVisibilityOnThreeDViews(isVisible):
    import slicer
    layoutManager = slicer.app.layoutManager()
    for i in range(layoutManager.threeDViewCount):
        threeDViewNode = layoutManager.threeDWidget(i).mrmlViewNode()
        threeDViewNode.SetBoxVisible(isVisible)
        threeDViewNode.SetAxisLabelsVisible(isVisible)


def setConventionalWideScreenView():
    import slicer
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView)
