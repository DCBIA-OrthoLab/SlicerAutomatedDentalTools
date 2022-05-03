import qt


class LMTab:
    def __init__(self,lm_dic) -> None:

        self.LM_tab_widget = qt.QTabWidget()
        self.LM_tab_widget.minimumSize = qt.QSize(100,200)
        self.LM_tab_widget.maximumSize = qt.QSize(800,400)

        cbd = {}
        for group,lm_lst in lm_dic.items():
            for lm in lm_lst:
                if lm not in cbd.keys():
                    cbd[lm] = qt.QCheckBox(lm)

        self.check_box_dic = cbd

        all_lm_tab = self.GenNewTab(self.check_box_dic.values())

        self.LM_tab_widget.insertTab(0,all_lm_tab,"All")

    def GenNewTab(self,widget_lst):
        new_widget = qt.QWidget()
        vb = qt.QVBoxLayout(new_widget)
        scr_box = qt.QScrollArea()
        vb.addWidget(scr_box)
        vb.addWidget(qt.QPushButton('Toggle'))

        wid = qt.QWidget()
        vb2 = qt.QVBoxLayout()
        for widget in widget_lst:
            vb2.addWidget(widget_lst)
        wid.setLayout(vb2)

        scr_box.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOn)
        scr_box.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        scr_box.setWidgetResizable(True)
        scr_box.setWidget(wid)

        return new_widget
        


