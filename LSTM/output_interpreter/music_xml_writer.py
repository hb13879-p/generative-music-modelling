import xml.etree.ElementTree as ET
import os

class MusicXmlWriter(object):
    def __init__(self, voicings, labels):
        self.voicing_list = voicings
        self.label_list = labels
        self.write_to_musicxml(self.voicing_list,self.label_list)


    def write(self,filename):
        self.tree.write(filename)

    def add_chord_symbol(self,root,bar,ch_root,quality,ch_root_alter=None,ch_degree_value=None,ch_degree_alter=None,ch_degree_type=None):
        measure = root.find(".//measure[@number='" + str(bar) + "']")
        backup = ET.SubElement(measure,"backup")
        duration = ET.SubElement(backup,"duration")
        duration.text = "1024"
        harmony = ET.SubElement(measure,"harmony")
        chord_root = ET.SubElement(harmony,"root")
        root_step = ET.SubElement(chord_root,"root-step")
        if ch_root_alter is not None:
            root_alter = ET.SubElement(chord_root,"root-alter")
            root_alter.text = str(ch_root_alter)
        kind = ET.SubElement(harmony,"kind")
        if ch_degree_value is not None and ch_degree_alter is not None and ch_degree_type is not None:
            degree = ET.SubElement(harmony,"degree")
            degree_value = ET.SubElement(degree,"degree-value")
            degree_alter = ET.SubElement(degree,"degree-alter")
            degree_type = ET.SubElement(degree,"degree-type")
            degree_value.text = str(ch_degree_value)
            degree_alter.text = str(ch_degree_alter)
            degree_type.text = ch_degree_type
        staff = ET.SubElement(harmony,"staff")
        staff.text = "1"
        kind.text = quality
        root_step.text = ch_root
        harmony.set("default-y","25")
        harmony.set("default-x","25")

    def add_note(self,measure,tone,tone_octave,default_x,count):
        note = ET.SubElement(measure,"note")
        note.set("color","#000000")
        note.set("default-x",str(default_x))
        if count > 0:
            chord = ET.SubElement(note,"chord")
        pitch = ET.SubElement(note,"pitch")
        step = ET.SubElement(pitch,"step")
        if len(tone) == 2:
            alter = ET.SubElement(pitch,"alter")
            if tone[1] == 'b':
                alter.text = "-1"
            elif tone[1] == "#":
                alter.text = "1"
            else:
                raise ValueError("Error in tone input")
        octave = ET.SubElement(pitch,"octave")
        octave.text = tone_octave
        step.text = tone[0]
        duration = ET.SubElement(note, "duration")
        duration.text = "1024"
        instrument = ET.SubElement(note, "instrument")
        instrument.set("id","P1-I1")
        voice = ET.SubElement(note, "voice")
        voice.text = "1"
        n_type = ET.SubElement(note, "type")
        n_type.text = "whole"
        if len(tone) == 2:
            accidental = ET.SubElement(note,"accidental")
            if tone[1] == "b":
                accidental.text = "flat"
            elif tone[1] == "#":
                accidental.text = "sharp"
            else:
                raise ValueError("Error in accidental input")
        staff = ET.SubElement(note,"staff")
        staff.text = "1"

    def add_chord(self,root,chord_tones,bar):
        measure = root.find(".//measure[@number='" + str(bar) + "']")
        note = measure.find("note")
        backup = measure.find("backup")
        measure.remove(note)
        measure.remove(backup)
        backup = ET.SubElement(measure,"backup")
        duration = ET.SubElement(backup,"duration")
        duration.text = "1024"
        for i,note in enumerate(chord_tones):
            self.add_note(measure,note[:-1],note[-1],86,i)

    def parse_label_info(self,label):
        root_alter = None
        if label[1] == 'b':
            quality = label[2:]
            root_alter = '-1'
        elif label[1] == '#':
            quality = label[2:]
            root_alter = '1'
        else:
            quality = label[1:]
        ch_root = label[0]
        quality_to_xml_input = {'diminished':['diminished',root_alter,None,None,None],'dominant':['dominant',root_alter,None,None,None],\
                'halfdim':['half-diminished',root_alter,None,None,None],'major':['major-seventh',root_alter,None,None,None],\
                'minor':['minor',root_alter,None,None,None],'majmin':['major-minor',root_alter,None,None,None],\
                'augmented':['augmented',root_alter,None,None,None],'domshp11':['dominant',root_alter,'11','1',"add"]}
        quality = quality_to_xml_input[quality]
        return str(ch_root),quality[0],quality[1],quality[2],quality[3],quality[4]

    def __insert_melody_xml(self,melody_xml,parent):
        for i,child in enumerate(parent):
            if "P1" in child.attrib.values():
                parent.remove(child)
                parent.insert(i,melody_xml)
        #parent.remove(part)
        #to_insert = ET.fromstring(melody_xml)
        #p1_root.append(to_insert)

    def write_to_musicxml(self,chords,chord_labels):
        if len(chords) != len(chord_labels):
            raise ValueError("Chords and chord labels not same length: chordlen = " + len(chords) + " labellen = " + len(chord_labels))
        self.tree = ET.parse(os.path.abspath("C:/Users/User/Documents/Project/MusicXML/plain_musicxml.xml"))
        root = self.tree.getroot()
        p1_root = root.find(".//part[@id='" + "P1" + "']")
        p2_root = root.find(".//part[@id='" + "P2" + "']")
        for i, label in enumerate(chord_labels):
            ch_root,quality,ch_root_alter,ch_degree_value,ch_degree_alter,ch_degree_type = self.parse_label_info(label)
            self.add_chord(p2_root,chords[i],i+1)
            self.add_chord_symbol(p2_root,i+1,ch_root,quality,ch_root_alter,ch_degree_value,ch_degree_alter,ch_degree_type)

    def insert_melody(self,melody_xml):
        root = self.tree.getroot()
        parent = root.find(".//score-partwise[@version='" + "3.0" + "']")
        self.__insert_melody_xml(melody_xml,root)

    #chords = [['A3','D4','E5','Ab4'],['A3','Eb4','E5','Ab4'],['A3','D4','G4','Ab4']]
    #chord_labels = ['Fdominant','Ebdomshp11','Cmajmin']
    #write_to_musicxml(chords, chord_labels)
