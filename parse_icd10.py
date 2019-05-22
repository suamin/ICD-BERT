# -*- coding: utf-8 -*-

import copy
import requests
import re
import os
import pickle as pkl

from bs4 import BeautifulSoup
from collections import defaultdict


class ICD10Hierarchy:
    
    def __init__(self):
        self.url = "https://www.dimdi.de/static/de/klassifikationen/icd/icd-10-gm/kode-suche/htmlgm2016/"
        self.build_tree()
        self.link_nodes()
    
    def build_tree(self):
        rget = requests.get(self.url)
        soup = BeautifulSoup(rget.text, "lxml")
        chapters = soup.findAll("div", {"class": "Chapter"})
        self.tree = dict()
        self.code2title = dict()
        
        def recurse_chapter_tree(chapter_elem):
            ul = chapter_elem.find("ul")
            codes = {}
            if ul is not None:
                # get direct child only
                ul = ul.find_all(recursive=False)
                for uli in ul:
                    uli_codes = recurse_chapter_tree(uli)
                    codes[uli.a.text] = {
                        "title": uli.a["title"],
                        "subgroups": uli_codes if uli_codes else None
                    }
                    self.code2title[uli.a.text] = uli.a["title"]
            return codes
        
        # used to clean chapter titles
        prefix_re = re.compile(r"Kapitel (?P<chapnum>[IVX]{1,5})") # I->minlen, XVIII->maxlen
        
        for chapter in chapters:
            # chapter code and title
            chap_h2 = chapter.find("h2").text[:-9]
            chap_code = chap_h2.strip("()")
            chap_title = prefix_re.sub("", chap_h2)
            chap_num = prefix_re.search(chap_h2).groupdict()['chapnum']
            if chap_num == "XIXV":
                # small fix for "XIXVerletzungen .." V is part of word
                chap_num = "XIX"
            # parse hierarchy
            self.tree[chap_num] = {
                "title": chap_title,
                "code": chap_code,
                "subgroups": recurse_chapter_tree(chapter)
            }
            self.code2title[chap_num] = chap_title
    
    def link_nodes(self):
        self.parent2childs = dict()
        
        def set_parent2childs(d):
            for k, v in d.items():
                if k not in ("title", "code", "subgroups"):
                    if v["subgroups"] is not None:
                        self.parent2childs[k] = set(v["subgroups"].keys())
                        set_parent2childs(v["subgroups"])
        
        set_parent2childs(self.tree)
        
        def update_parent2childs():
            parent2childs = copy.deepcopy(self.parent2childs)
            
            def get_all_descendants(parent, childs):
                temp_childs = copy.deepcopy(childs)
                for childi in temp_childs:
                    # get child's childs
                    if childi in parent2childs:
                        # recurse till leaf nodes
                        get_all_descendants(childi, parent2childs[childi])
                        parent2childs[parent].update(parent2childs[childi])
            
            for parent, childs in self.parent2childs.items():
                get_all_descendants(parent, childs)
            
            self.parent2childs = parent2childs
        
        update_parent2childs()
        
        # get reversed mapping
        self.child2parents = defaultdict(set)
        
        for parent, childs in self.parent2childs.items():
            for childi in childs:
                self.child2parents[childi].add(parent)


if __name__=="__main__":
    icd10hier = ICD10Hierarchy()
    num_chaps = len(icd10hier.tree)
    num_par = len(icd10hier.parent2childs)
    num_ch = len(icd10hier.child2parents)
    total = len(icd10hier.code2title)
    print(
        "[INFO] found %d ICD-10-GM-v2016 codes (%d chapters | %d parents | %d children) in hierarchy" 
        % (total, num_chaps, num_par, num_ch)
    )
    print("[INFO] src: %s" % icd10hier.url)
    os.makedirs("tmp", exist_ok=True)
    with open(os.path.join("tmp", "icd10gmv2016_hier.pkl"), "wb") as wf:
        pkl.dump(icd10hier, wf)
