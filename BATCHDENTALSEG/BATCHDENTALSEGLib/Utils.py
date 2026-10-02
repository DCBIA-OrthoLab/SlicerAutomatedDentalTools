import qt


def createButton(name, callback=None, is_checkable=False, icon=None, toolTip="", parent=None):
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
    button.setCheckable(is_checkable)
    button.setToolTip(toolTip)
    return button


def addInCollapsibleLayout(child_widget, parent_layout, collapsible_text, isCollapsed=True):
    """
    Wraps input childWidget into a collapsible button attached to input parentLayout.
    collapsibleText is writen next to collapsible button. Initial collapsed status is customizable
    (collapsed by default)
    """
    import ctk
    collapsible_button = ctk.ctkCollapsibleButton()
    collapsible_button.text = collapsible_text
    collapsible_button.collapsed = isCollapsed
    parent_layout.addWidget(collapsible_button)
    collapsible_button_layout = qt.QVBoxLayout()
    collapsible_button_layout.addWidget(child_widget)
    collapsible_button.setLayout(collapsible_button_layout)


def set3DViewBackgroundColors(top_color, bottom_color):
    """ Set the background color as a gradient between the top and bottom colors

    :param topColor: (r, g, b) floats between 0 and 1
    :param bottomColor: (r, g, b) floats between 0 and 1
    """
    import slicer
    view_node = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
    view_node.SetBackgroundColor(bottom_color)
    view_node.SetBackgroundColor2(top_color)


def setBoxAndTextVisibilityOnThreeDViews(is_visible):
    import slicer
    layout_manager = slicer.app.layoutManager()
    for i in range(layout_manager.threeDViewCount):
        three_d_view_node = layout_manager.threeDWidget(i).mrmlViewNode()
        three_d_view_node.SetBoxVisible(is_visible)
        three_d_view_node.SetAxisLabelsVisible(is_visible)


def setConventionalWideScreenView():
    import slicer
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView)
