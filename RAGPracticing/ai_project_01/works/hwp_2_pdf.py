import os
import re
import win32com.client


# params
hwp_path = 'C:\\Users\\qksso\\Downloads\\dataset\\under30'
pdf_path = 'C:\\Users\\qksso\\Downloads\\dataset\\under30'

try:
    os.mkdir(hwp_path)
except:
    pass

try:
    os.mkdir(pdf_path)
except:
    pass

# load hwp api
hwp = win32com.client.gencache.EnsureDispatch('HWPFrame.HwpObject')
hwp.RegisterModule('FilePathCheckDLL', 'FilePathCheckerModule')

file_list = [f for f in os.listdir(hwp_path) if re.match('.*[.](hwp|hwpx)', f)]
for file in file_list:
    pre, ext = os.path.splitext(file)

    src = f'{hwp_path}/{file}'
    dst = f'{pdf_path}/{pre + ".pdf"}'

    # open hwp file
    hwp.Open(src)
    # set save filename
    hwp.HParameterSet.HFileOpenSave.filename = dst
    # set save format to "pdf"
    hwp.HParameterSet.HFileOpenSave.Format = "PDF"
    # save
    hwp.HAction.Execute("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
    print(f'convert {src} to {dst}')
hwp.Quit()