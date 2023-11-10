import xml.etree.ElementTree as ET
import os
import sys

def getFilenames(path):
    return os.listdir(path)

def getContent(path,xml_filename):
    if xml_filename[-4:] != ".xml":
        xml_filename += ".xml"
    tree = ET.parse(os.path.join(path,xml_filename))
    root = tree.getroot()

    content_obj = root.find("Content")
    if content_obj is not None:
        content = content_obj.text
    else:
        content = "None"
    species_obj = root.find("Species")
    if species_obj is not None:
        species = species_obj.text
    else:
        species = "None"
    return content,species

def moveFile(filePath,outdir):
    os.rename(filePath, os.path.join(outdir,os.path.basename(filePath)))

def moveRecord(indir,name,outdir):
    xml = os.path.join(indir,name + ".xml")
    jpg = os.path.join(indir,name + ".jpg")
    png = os.path.join(indir,name + ".png")
    if os.path.isfile(xml) and (os.path.isfile(jpg) or os.path.isfile(png)):
        moveFile(xml,outdir)
        try:
            if os.path.isfile(jpg):
                moveFile(jpg,outdir)
            else:
                moveFile(png,outdir)
        except KeyboardInterrupt as e: #atomicness just bodged in
            if os.path.isfile(jpg):
                moveFile(jpg,outdir)
            elif os.path.isfile(png):
                moveFile(png,outdir)
            raise e
        return True
    else:
        return False

def binSearch(item,sorted_list,start=0,end=-1):
    if end == -1:
        end = len(sorted_list) - 1
    if start < 0 or start >= len(sorted_list):
        raise IndexError("start index outside range")
    elif end < 0 or end >= len(sorted_list):
        raise IndexError("end index outside range")
    elif start > end:
        raise IndexError("start index cannot be greater than end index")
    return _binSearch(item,sorted_list,start,end)
def _binSearch(item,sorted_list,st,en):
    if st > en:
        return False
    mid = st + (en - st)//2
    if item < sorted_list[mid]:
        return _binSearch(item,sorted_list,st,mid - 1)
    elif item > sorted_list[mid]:
        return _binSearch(item,sorted_list,mid + 1,en)
    else:
        return True

def binIndex(item,sorted_list,start=0,end=-1):
    if end == -1:
        end = len(sorted_list) - 1
    if start < 0 or start >= len(sorted_list):
        raise IndexError("start index outside range")
    elif end < 0 or end >= len(sorted_list):
        raise IndexError("end index outside range")
    elif start > end:
        raise IndexError("start index cannot be greater than end index")
    return _binIndex(item,sorted_list,start,end)
def _binIndex(item,sorted_list,st,en):
    if st > en:
        return None
    mid = st + (en - st)//2
    if item < sorted_list[mid]:
        return _binIndex(item,sorted_list,st,mid - 1)
    elif item > sorted_list[mid]:
        return _binIndex(item,sorted_list,mid + 1,en)
    else:
        return mid

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("dataset_filter.py inputdir outputdir [speciesListFile]")
        sys.exit()
    inp = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    if len(sys.argv) == 4:
        species_file = os.path.abspath(sys.argv[3])
        if not os.path.isfile(species_file):
            raise OSError("Invalid species file: " + species_file)
        elif not os.access(species_file,os.R_OK):
            raise OSError("No read permission for species file: " + species_file)
        with open(species_file) as f:
            species_list = f.read().splitlines()
            print("Looking for " + str(len(species_list)) + " species")

    if not os.path.isdir(inp):
        raise OSError("Invalid input directory: " + inp)
    elif not os.access(inp,os.R_OK):
        raise OSError("No read permission for input directory: " + inp)
    if not os.path.isdir(out):
        raise OSError("Invalid output directory: " + out)
    elif not os.access(out,os.W_OK):
        raise OSError("No write permission for output directory: " + inp)

    filenames = getFilenames(inp)
    num_files = str(len(filenames))
    cur = 0
    total = 0
    try:
        for name in filenames:
            cur += 1
            if name[0] != "_" and name[-4:] == ".xml":
                content, species = getContent(inp,name)
                if content == "Entire":
                    if len(sys.argv) != 4:
                        total += moveRecord(inp,name[:-4],out)
                    elif binSearch(species,species_list): #need to look at species
                        total += moveRecord(inp,name[:-4],out)
            print(str(cur) + "/" + num_files + " processed, " + str(total) + " applicable", end='\r')
    except KeyboardInterrupt:
        print("")
        print("Keyboard Interrupt, stopping...")
    print(str(total) + " applicable records found and moved")

if __name__ == "__main__":
    main()