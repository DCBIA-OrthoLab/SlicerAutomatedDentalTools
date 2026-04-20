import pandas as pd
import itertools


class Group_landmark:
    """
    Manage list landmark usefull for user

    This class it create to make it easier to found information in group_landmark's user

    exemple :
    {'Mandible': MyList(suffix=['RAF', 'LAF', 'RAE', 'LAE', 'Rco', 'Lco', 'RLCo', 'RMCo', 'LLCo', 'LMCo', 'RSig', 'LSig', 'RPRa', 'LPRa', 'RARa', 'LARa', 'RGo', 'LGo', 'RMe', 'LMe', 'PogL', 'B', 'Pog ', 'Gn', 'Me']),
     'Maxilla': MyList(suffix=['ROr', 'LOr', 'RInfOr', 'LInfOr', 'RMZyg', 'LMZyg', 'RNC', 'LNC', 'RPF', 'LPF', 'IF', 'ANS', 'PNS', 'A']),
     'Cranial base/Cervical vertebra': MyList(suffix=['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4']),
     'Dental': MyDict(suffix=['R', 'O', 'MB', 'DB', 'OIP', 'RIP', 'MBIP', 'DBIP', 'CB', 'CL', 'CBIP', 'CLIP', 'RC'], prefix={'Lower': ['LR7', 'LR6', 'LR5', 'LR4', 'LR3', 'LR2', 'LR1', 'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LL7'], 'Upper': ['UR7', 'UR6', 'UR5', 'UR4', 'UR3', 'UR2', 'UR1', 'UL1', 'UL2', 'UL3', 'UL4', 'UL5', 'UL6', 'UL7']}),
     'Other': MyList(suffix=['Pog']),
     'Midpoint': MyList(suffix=['Mid_ROr_LOr', 'Mid_RCo_LCo', 'Mid_RMZyg_LMZyg', 'Mid_RGo_LGo', 'Mid_RLCo_LLCo', 'Mid_RMCo_LMCo'])}


    how to use :
        - init : need excel path

            with this architecture below, Group_landmark's init make the exemple above

   Mandible Maxilla Cranial base/Cervical vertebra Dental Dental-Lower Dental-Upper
0       RAF     ROr                             Ba      R          LR7          UR7
1       LAF     LOr                              S      O          LR6          UR6
2       RAE  RInfOr                              N     MB          LR5          UR5
3       LAE  LInfOr                            RPo     DB          LR4          UR4
4       Rco   RMZyg                            LPo    OIP          LR3          UR3
5       Lco   LMZyg                          RFZyg    RIP          LR2          UR2
6      RLCo     RNC                          LFZyg   MBIP          LR1          UR1
7      RMCo     LNC                             C2   DBIP          LL1          UL1
8      LLCo     RPF                             C3     CB          LL2          UL2
9      LMCo     LPF                             C4     CL          LL3          UL3
10     RSig      IF                            NaN   CBIP          LL4          UL4
11     LSig     ANS                            NaN   CLIP          LL5          UL5
12     RPRa     PNS                            NaN     RC          LL6          UL6
13     LPRa       A                            NaN    NaN          LL7          UL7
14     RARa     NaN                            NaN    NaN          NaN          NaN
15     LARa     NaN                            NaN    NaN          NaN          NaN
16      RGo     NaN                            NaN    NaN          NaN          NaN
17      LGo     NaN                            NaN    NaN          NaN          NaN
18      RMe     NaN                            NaN    NaN          NaN          NaN
19      LMe     NaN                            NaN    NaN          NaN          NaN
20     PogL     NaN                            NaN    NaN          NaN          NaN
21        B     NaN                            NaN    NaN          NaN          NaN
22     Pog      NaN                            NaN    NaN          NaN          NaN
23       Gn     NaN                            NaN    NaN          NaN          NaN
24       Me     NaN                            NaN    NaN          NaN          NaN



        - contain : to check if landmark exist in group_landmark, you just need to do this : 'LD' in Group_Landmark

        - items : same utility as dict




    """

    def __init__(self, path_listlandmarks) -> None:
        self.group_landmark = dict()
        reader = pd.read_excel(path_listlandmarks)
        header_before = "b suisv"  # random str

        for header in reader.keys():

            if header_before in header:

                header2 = header.split("-")[1]
                Type = self.group_landmark[header_before]
                tmplist = []
                for landmark in reader[header].tolist():
                    if isinstance(landmark, str):
                        tmplist.append(landmark)
                self.group_landmark[header_before] = Type.add({header2: tmplist})

            else:
                header_before = header
                tmplist = []
                for landmark in reader[header].tolist():
                    if isinstance(landmark, str):
                        tmplist.append(landmark)

                self.group_landmark[header] = MyList(tmplist)

    def __repr__(self):
        return f"{self.group_landmark}"

    def __contains__(self, landmark):
        out = False
        for landmarks in self.group_landmark.values():
            if landmark in landmarks:
                out = True
                break

        return out

    def existsInDict(self, list_landmark):
        """return a dict with key the name of landmark and whth value False or True according to if landmark is in group landmark

        exitsInDict is different that __contain__.
        Exemple : __contain__('LL6O') == True
                exitsInDict(['LL6O']) == {'LL6' : True, 'O' : True}

        Args:
            list_landmark (list): landmkar's list

        Returns:
            dict: return a dict with key the name of landmark and in value False or True according to if landmark is in group landmark
        """
        dic_out = {key: False for key in self.tolist()}

        for landmark in list_landmark:
            if self.__contains__(landmark):
                dic_out.update(self.existInDict(landmark))

        return dic_out

    def existInDict(self, landmark: str):
        """ find function is the same methode that existsInDict but with str input and not list

        Args:
            landmark (str): _description_

        Returns:
            dict: _description_
        """
        out = {landmark: False}
        for type in self.group_landmark.values():
            if landmark in type:
                out = type.existInDict(landmark)
                break
        return out

    def tolist(self):
        out_list = []
        for values in self.group_landmark.values():
            out_list += values.tolist()

        return out_list

    def __setitem__(self, key, value):
        if isinstance(value, list):
            self.group_landmark[key] = MyList(value)

        elif isinstance(value, str):
            self.group_landmark[key].set(value)

    def __getitem__(self, key):
        return self.group_landmark[key]

    def items(self):
        """_summary_

        Returns:
            tuples, tuples : return keys and value of group landmark
        """
        return self.group_landmark.items()

    def keys(self):
        return self.group_landmark.keys()


class MyList:
    def __init__(self, suffix: list) -> None:
        self.suffix = suffix

    def add(self, dic):
        return MyDict(suffix=self.suffix, prefix=dic)

    def __contains__(self, landmark):
        return landmark.upper() in [lm.upper() for lm in self.suffix]

    def existInDict(self, landmark):
        out = False
        if self.__contains__(landmark):
            out = True

        return {landmark: out}

    def __iter__(self):
        self.iter = -1
        return self

    def __next__(self):
        if self.iter + 1 >= len(self.suffix):
            raise StopIteration
        self.iter += 1
        return self.suffix[self.iter]

    def set(self, value):
        self.suffix.append(value)

    def tolist(self):
        return self.suffix

    def __iadd__(self,__o : list):
        self.suffix += __o


class MyDict(MyList):
    def __init__(self, suffix: list, prefix: dict) -> None:
        super().__init__(suffix)
        self.prefix = prefix

    def __contains__(self, landmark):

        out = False
        pre, suf = self.decomp(landmark)
        if not None in (pre, suf):
            out = True
        return out

    def decomp(self, landmark: str):
        pre = None
        suf = None
        for prefix in list(itertools.chain.from_iterable(self.prefix.values())):
            if prefix.upper() == landmark[: len(prefix)].upper():
                for suffix in self.suffix:
                    if suffix.upper() == landmark[len(prefix) :].upper():
                        pre = prefix.upper()
                        suf = suffix.upper()
                        break
        return pre, suf

    def existInDict(self, landmark: str):
        pre, suf = self.decomp(landmark)
        out = {landmark: False}
        if not None in (pre, suf):
            out = {pre: True, suf: True}

        return out

    def add(self, dic):
        self.prefix.update(dic)
        return self

    def getSeparatePreSuf(self):
        return self.prefix.copy(), self.suffix.copy()

    def tolist(self):
        return self.suffix + list(itertools.chain.from_iterable(self.prefix.values()))
