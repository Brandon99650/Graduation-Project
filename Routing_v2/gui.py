import os
from functools import partial
from utils.dicttool import *
import numpy as np
import pandas as pd

from diversityrouting import DiversityRouter

from utils.attractiondata import AttractionDataset
from kivy.metrics import dp
from kivy.resources import resource_find
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView


downward_arrow = str(u'\u2193')

KVcolormap = {
    'red':(1,0,0), 'green':(0,1,0), 'blue':(0,0,1),
    'yellow': (1,1,0), 'white':(1,1,1), 'black':(0,0,0)
}

label_text = [
    u"自然探索", u"山野農情", u"文化信仰",
    u"藝術人文", u"海湖風光", u"歷史紀念"
]
db = AttractionDataset(
    os.path.join("Attractiondata", "cluster_with_coo.csv")
)
router = DiversityRouter(
    dbpath=os.path.join("Attractiondata", "cluster_with_coo.csv")
)


class APPScreens(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.screenstack = []
        self.condition = {
            'label':None,
            'region':None
        }

Whole_screen_manager = APPScreens()


def homepage(instance):
    Whole_screen_manager.current = Whole_screen_manager.screenstack[0]
    for i in Whole_screen_manager.screenstack[1:]:
        Whole_screen_manager.remove_widget(
            Whole_screen_manager.get_screen(name=i)
        )
    Whole_screen_manager.screenstack=Whole_screen_manager.screenstack[:1]

class Label_selection(GridLayout):
    def __init__(self, label_list:list, **kwargs):
        super(Label_selection, self).__init__(**kwargs)
        self._select_label = list(0 for _ in range(len(label_list)))
        self._button_list = self._gen_button_list(label_list)
        self.height= dp(800)
        self.rows = len(self._button_list)//2
        self.cols = 2
        self._add_button_list(self._button_list)
        
    
    def _gen_button_list(self, label_text):
        class_button_lst = []
        for i, label in enumerate(label_text):
            b = Button(
                text=label,font_size = dp(20),
                background_color = KVcolormap['white'],
                font_name = resource_find('ukai.ttc'),
                color = KVcolormap['white']
            )
            b.bind(on_press = partial(self.press, i))
            class_button_lst.append(b)
        return class_button_lst

    def _add_button_list(self, button_list):
        for b in button_list:
            self.add_widget(b)
        
    def press(self,idx,instance):
        if self._select_label[idx] == 1:
            self._select_label[idx] = 0
            instance.color = KVcolormap['white']
            
        else:
            self._select_label[idx] = 1
            instance.color = KVcolormap['yellow']

def gen_sources(region_conditon, wanted_label, debug=True):

    k_ = ["region","town"]
    in_region = db.df.copy()
    for k,r in zip(k_, region_conditon):
        in_region = in_region[in_region[k]==r]
    if debug:
        in_region.to_csv("buf.csv")
    if in_region.shape[0] == 0:
        return None
    
    their_label = in_region[list(str(i) for i in range(6))].values
    expect_label = np.tile(
        np.array(wanted_label), (their_label.shape[0],1)
    )
    rms = np.sum((expect_label-their_label)**2, axis=1)
    fitting = np.where(rms == rms.min())[0]
    fitting_attraction = in_region.iloc[fitting]
    return fitting_attraction

class SourceSelection(ScrollView):
    
    def __init__(self,t, fitting_attraction:pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.do_scroll_x: False
        self.do_scroll_y: True
        self.t = t
        self.inside = GridLayout()
        self.goback = Button(
            text="返回",
            font_name = resource_find('ukai.ttc'),
            font_size = 20, 
            background_color = KVcolormap['blue']
        )
        self.home_b = Button(
            text="首頁",
            font_name = resource_find('ukai.ttc'),
            font_size = 20, 
            background_color = KVcolormap['blue']
        )
        self.home_b.bind(on_release=homepage)
        self.goback.bind(on_release=self.back)

        self.last = None
        self.inside.rows = fitting_attraction.shape[0]+2
        self.inside.cols = 2
        if fitting_attraction.shape[0] < 5:
            self.inside.size_hint = (1.0, 0.3)
        else:
            self.inside.size_hint = (1.0, 0.8)
        
     
        self.candidate = fitting_attraction
        self.label_list = self._gen_label_list(
            self.candidate[list(str(i) for i in range(6))].values.tolist()
        )
        self.button_list = self._gen_button_list(
            self.candidate.name.tolist()
        )

        self.inside.add_widget(self.home_b)
        self.inside.add_widget(self.goback)

        self._add_button_label_list(
            self.button_list, self.label_list
        )
        self.ok = Button(text="OK")
        self.ok.bind(on_release=self.submit)
        self.inside.add_widget(Label(text=""))
        self.inside.add_widget(self.ok)
        self.add_widget(self.inside)
        self.thesource = None
    
    def _gen_label_list(self, label_):
        l = []
        for t in label_:
            itslabel = ""
            for idx, ti in enumerate(t):
                if ti > 0:
                    itslabel = f"{itslabel}{label_text[idx]},"
            l.append(Label(
                text=itslabel[:-1],
                font_name = resource_find('ukai.ttc'),
                font_size = 20, 
                
                )
            )
        return l

    def  _gen_button_list(self, label_text):
        btlist = []
        for i, label in enumerate(label_text):
            bi = Button(
                text=label,font_size = dp(20),
                background_color = KVcolormap['white'],
                font_name = resource_find('ukai.ttc'),
                color = KVcolormap['white']
            )
            bi.bind(on_release=partial(self.press, i))
            btlist.append(bi)
        
        return btlist
    
    def _add_button_label_list(self, button_list,label_list):
        for b, l in zip(button_list, label_list):
            self.inside.add_widget(b)
            self.inside.add_widget(l)            

    def press(self, idx, instance):
        self.thesource=self.candidate.iloc[idx][['name','placeid']]
        instance.background_color = KVcolormap['yellow']
        if self.last is not None:
            self.button_list[self.last].background_color=KVcolormap['white']
        self.last = idx

        self.ok.background_color = KVcolormap['blue']
    
    def back(self, instance):
        Whole_screen_manager.current = "selection"
        Whole_screen_manager.condition['label']=None
        Whole_screen_manager.condition['region']=None
        Whole_screen_manager.screenstack=Whole_screen_manager.screenstack[:-1]
        Whole_screen_manager.remove_widget(
            Whole_screen_manager.get_screen("sources")
        )
    def submit(self, instance):
        if self.last is not None:
            #print(self.thesource)
            pass
        name = self.thesource['name']
        s= Screen(name=name)
        s.add_widget(
            ShowPath(
                self.thesource['placeid'],self.t, 
                fromwho="sources"
            )
        )
        Whole_screen_manager.add_widget(s)
        Whole_screen_manager.screenstack.append(name)
        Whole_screen_manager.current = name
    
class ShowPath(GridLayout):
    
    def __init__(self, srcid,t, fromwho, **kwargs):
        super().__init__(**kwargs)
        self.rows = 8
        self.cols = 2
        self.fromwho = fromwho
        self.sname=None
        self.path = self._routing(sourceid=srcid, t=t)
        self._build_layout()

        self.goback=Button(
            text="返回",
            font_name = resource_find('ukai.ttc'),
            font_size = 20, 
            background_color = KVcolormap['blue'],
            size_hint=(0.4,0.2)
        )
        
        self.goback.bind(on_release = self._back)
        self.add_widget(self.goback)
        self.add_widget(
            Label(
                text=f"score : {self.path['eva'][1]}",
                size_hint=(0.4,0.2)
            )
        )
        
    def _routing(self, sourceid,t):
        return router.routing(
            sourceID=sourceid,debug=False,timelimit=t
        )
    
    def _build_layout(self):
        self.sname= self.path['attrname'][0]
        #print(self.sname)
        label_list = []
        l_list = []
        for attri_name, attri_label in zip(
            self.path['attrname'],self.path['label'],
        ):
            l_ = "["
            for lidx, li in enumerate(attri_label):
                if li>0:
                    l_ = f"{l_}{label_text[lidx]},"
            label_list.append(attri_name)
            l_list.append(f"{l_[:-1]}]")

        i = 0
        for attr, li in zip(label_list, l_list):
            attr_b = Button(
                text=attr, 
                font_name = resource_find('ukai.ttc'),
                font_size = 20, size_hint=(0.4,0.2)
            )
            attr_b.bind(
                on_release=partial(self._each_des, i)
            )
            self.add_widget(attr_b)
            self.add_widget(
                Label(
                    text=li, 
                    font_name = resource_find('ukai.ttc'),
                    font_size = 20, size_hint=(0.4,0.2),
                    pos_hint={'right':0.5}
            ))
            if i < len(label_list)-1:
                self.add_widget(
                    Label(
                    text=f"{downward_arrow} {self.path['travel_time'][i]:.0f} mins", 
                    font_name = resource_find('ukai.ttc'),
                    font_size = 20, size_hint=(0.4,0.2),
                    pos_hint={'right':0.5}
                ))

                self.add_widget(
                    Label(
                    text="", 
                    font_name = resource_find('ukai.ttc'),
                    font_size = 20, size_hint=(0.4,0.2),
                    pos_hint={'right':0.5}
                ))
            i += 1
   
    def _each_des(self, idx, instance):
        s = Screen(name='des')
        s.add_widget(
            Description(
                itsname= self.path['attrname'][idx],
                addr=self.path['address'][idx],
                des=self.path['attr_desc'][idx], 
                top_s=self.sname)
            )
        Whole_screen_manager.add_widget(s)
        Whole_screen_manager.current = 'des'

    def _back(self, instance):
        Whole_screen_manager.current = self.fromwho
        Whole_screen_manager.screenstack=Whole_screen_manager.screenstack[:-1]
        Whole_screen_manager.remove_widget(
            Whole_screen_manager.get_screen(name=self.sname)
        )


class Description(FloatLayout):
    
    def __init__(self, itsname,addr, des, top_s, **kwargs):
        super().__init__(**kwargs)
   
        self.top_s = top_s
       
        self.goback=Button(
            text="返回",
            font_name = resource_find('ukai.ttc'),
            font_size = 16, 
            background_color = KVcolormap['blue'],
            size_hint=(0.1,0.1),
            pos_hint = {"right":0.1, "top":1}
        )
        self.goback.bind(on_release=self._back)
        self.add_widget(self.goback)
        
        self.add_widget(
            Label(
                text=f"{itsname} 簡介 : \n{addr}",
                font_name = resource_find('ukai.ttc'),
                font_size = 25, 
                size_hint=(0.5,0.1),
                pos_hint={'right':0.8,'top':1}
            )
        )
        self.add_widget(
            Label(
                text=self._split_des(des, max_one_line=30),
                font_name = resource_find('ukai.ttc'),
                font_size = 23, 
                size_hint=(1,0.8),
                pos_hint={'right':1.0, 'top':0.8}
            )
        )
    
    def _back(self, instance):
        Whole_screen_manager.current = self.top_s
        Whole_screen_manager.remove_widget(
            Whole_screen_manager.get_screen("des")
        )
    
    def _split_des(self, des, max_one_line = 30):
        f = ""
        for i, char in enumerate(des):
            f = f"{f}{char}"
            if i%max_one_line == 0 and i > 0:
                f = f"{f}\n"
        return f


class CountryCity_selection(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_selection = []

region_selector =  CountryCity_selection(
    size_hint = (0.5, 0.4),
    pos_hint ={"right":0.52,"top":0.5}
)

class Subregion(GridLayout):
    def __init__(self, regionlist, sname,**kwargs):
        super().__init__(**kwargs)
        self.sname = sname
        self.rows = 5
        self.rnum=len(regionlist)
        self.last= None
        self.cols = self.rnum//self.rows + 1
        self.region_button = self._gen_button_list(regionlist)
        self._add_button_list(self.region_button)
    
    def _gen_button_list(self, text_list)->list:
        button_list = []
        for i, text_ in enumerate(text_list):
            #print(text_)
            b = Button(
                text=text_,font_size = 16,
                background_color = KVcolormap['white'],
                font_name = resource_find('ukai.ttc'),
                color = KVcolormap['white']
            )
            b.bind(on_release = partial(self.press,i, text_))
            button_list.append(b)
        b = Button(
            text="返回",font_size = 16,
            background_color = KVcolormap['blue'],
            font_name = resource_find('ukai.ttc'),
            color = KVcolormap['white']
        )
        b.bind(on_release=self.back)
        button_list.append(b)
        return button_list
    
    def _add_button_list(self, button_list):
        for b in button_list:
            self.add_widget(b)
        
    def back(self, instance):
        region_selector.current= "main"
        region_selector.current_selection = []
        region_selector.remove_widget(
            region_selector.get_screen(self.sname)
        )
        #print(region_selector.current_selection)
    
    def press(self, i, name, instance):
        instance.background_color = KVcolormap['yellow']
        if self.last is not None:
            self.last.background_color = KVcolormap['white']
        self.last = instance
        self.selection_region = name
        region_selector.current_selection =[self.sname, name]
        #print(region_selector.current_selection)

class CountryCity(GridLayout):
    
    def __init__(self, region:dict,**kwargs):
        super(CountryCity, self).__init__(**kwargs)
        self._setting = kwargs
        self.rows = 5
        self.cols = len(region.items())//self.rows + 1 
        self.tw_city_country = region
    
        self._countrycity_button = self._gen_countrycity_boutton(
            countrycity_list=self.tw_city_country
        )
        self.selection_region = None
        self.current_selection = None
        self._add_button_list(self._countrycity_button)
    
    def _gen_countrycity_boutton(self, countrycity_list:dict):
        button_list = []
        root_rigt = 0.0
        root_top = 2.0
        for i, item in enumerate(countrycity_list.items()):
            
            b = Button(
                text=item[0],font_size = 20,
                background_color = KVcolormap['white'],
                font_name = resource_find('ukai.ttc'),
                color = KVcolormap['white'],
                height= dp(800)
            )
     
            b.bind(on_release = partial(self.press, item[0]))
            button_list.append(b)
        return button_list
    
    def _add_button_list(self, button_list):
        for b in button_list:
            self.add_widget(b)
        
    def press(self, cname, instance):
        #print(f"press {cname}")
        tmp = Screen(name=cname)
        subregion_window = Subregion(
            regionlist=self.tw_city_country[cname],
            sname=cname,
            **self._setting
        )
        tmp.add_widget(subregion_window)
        region_selector.add_widget(tmp)
        region_selector.current= cname

class exception(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goback = Button(
            text="此區目前無資料!\n返回",
            font_size = 30,
            background_color = KVcolormap['blue'],
            font_name = resource_find('ukai.ttc'),
            color = KVcolormap['yellow'],
            size_hint=(1,1),
            pos_hint={'right':1, 'top':1}
        )
        self.goback.bind(on_release=self._back)
        self.add_widget(self.goback)
    
    def _back(self, instance):
        Whole_screen_manager.current='selection'
        Whole_screen_manager.remove_widget(
            Whole_screen_manager.get_screen("exception")
        )


class HomeLayout(FloatLayout):

    def __init__(self, dbpath, **kwargs):
        super(HomeLayout, self).__init__(**kwargs)

        self.tw_region =loadjson('tw_countrycity.json')
    
        self._build_label_selection()
        self._build_city_selection()
        self._build_submit_botton()
        self.timelimit = 30
        self._time_limit_input()
        self.source = None
        self._assign_source_input()


    def _build_label_selection(self):
        #Choose source label  
        self.label_selection_icon = Label(
            text="標籤",
            font_name = resource_find('ukai.ttc'),
            font_size = 30, 
            size_hint = (0.5, 0.1), 
            pos_hint ={"right":0.52,"top":1.0}
        )
        self.add_widget(self.label_selection_icon)

        self.label_selecting_section = Label_selection(
            label_list=label_text,
            size_hint = (0.5, 0.3), 
            pos_hint ={"right":0.52,"top":0.9}
        )
        self.add_widget(self.label_selecting_section)

    def _build_city_selection(self):
        #Choose source country/city 
        self.countries_cities_selection_icon = Label(
            text="城市",font_name = resource_find('ukai.ttc'),
            font_size = 30, 
            size_hint = (0.5, 0.1), 
            pos_hint ={"right":0.52,"top":0.6}
        )
        self.add_widget(self.countries_cities_selection_icon)

        region_selector.add_widget(self._build_country_city_mainpage())        
        self.add_widget(region_selector)

    def _build_submit_botton(self):
        #submit button
        self.submit_button = Button(
            text="GO!",font_size = 20,
            background_color = KVcolormap['blue'],
            color = KVcolormap['white'],
            size_hint = (0.5, 0.1),
            pos_hint ={"right":0.7,"top":0.1}
        )
        self.submit_button.bind(on_press=self.submit)
        self.add_widget(self.submit_button)

    def submit(self, instance):
        
        if len(self.timelimit_input.text):
            self.timelimit = int(self.timelimit_input.text)

        if len(self.directly_src.text):
            src_d = self.directly_src.text
            if not src_d in db.df.name.tolist():
                pass
            else:
                s = Screen(name=src_d)
                placeid = db.df[db.df['name']==src_d]['placeid'].values[0]
                s.add_widget(
                    ShowPath(placeid,self.timelimit,fromwho="selection")
                )
                Whole_screen_manager.add_widget(s)
                Whole_screen_manager.current = src_d
            
            return 

        if not len(region_selector.current_selection):
            print("NONO")
            return
        
        Whole_screen_manager.condition['label']=self.label_selecting_section._select_label
        Whole_screen_manager.condition['region'] = region_selector.current_selection
        
        srcs = gen_sources(
            Whole_screen_manager.condition['region'],
            Whole_screen_manager.condition['label'],
            debug=True
        )
        if srcs is None:
            s = Screen(name="exception")
            s.add_widget(exception())
            Whole_screen_manager.add_widget(s)
            Whole_screen_manager.current="exception"
        else:
            src_set =Screen(name="sources")
            src_set.add_widget(
                SourceSelection(
                    t= self.timelimit,
                    fitting_attraction=srcs
                )
            )
            Whole_screen_manager.add_widget(src_set)
            Whole_screen_manager.current = "sources"
            Whole_screen_manager.screenstack.append("sources")

    def _time_limit_input(self):
        self.timelimit_label = Label(
            text="兩地時限(min):\ndefault:30",
            font_name = resource_find('ukai.ttc'),
            font_size = 20, 
            size_hint = (0.2, 0.1), 
            pos_hint ={"right":0.75,"top":0.9}
        )
        self.timelimit_input = TextInput(
            multiline=False,
            size_hint = (0.15, 0.05),
            pos_hint ={"right":0.9,"top":0.9}
        )
        self.add_widget(self.timelimit_label)
        self.add_widget(self.timelimit_input)

    def _assign_source_input(self):
        self.src_choose_label = Label(
            text="直接指定起點:",
            font_name = resource_find('ukai.ttc'),
            font_size = 20, 
            size_hint = (0.2, 0.1), 
            pos_hint ={"right":0.75,"top":0.6}
        )
        self.add_widget(self.src_choose_label)
        self.directly_src = TextInput(
            multiline=False,font_name = resource_find('ukai.ttc'),
            size_hint = (0.3, 0.05),
            pos_hint ={"right":0.86,"top":0.5}
        )
        self.add_widget(self.directly_src)

    def _build_country_city_mainpage(self)->Screen:
        m = CountryCity(region=self.tw_region)
        s = Screen(name="main")
        s.add_widget(m)
        return s
    



class GUIapp(App):

    def build(self):
        Whole_screen_manager.add_widget(
            self._build_selection_page()
        )
        Whole_screen_manager.screenstack.append("selection")
        return Whole_screen_manager
    
    def _build_selection_page(self):
        selection = Screen(name="selection")
        selection.add_widget(
            HomeLayout(
                dbpath=os.path.join(
                    "Attractiondata", "cluster_with_coo.csv"
                )
            )
        )
        return selection



def main():
    gui = GUIapp()
    gui.run()
    #p = getpath()
    #print(p)

if __name__ == "__main__":
    main()
        
        