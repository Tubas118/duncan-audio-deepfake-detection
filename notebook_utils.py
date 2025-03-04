import jupytext

global_lastCell = 'top'

# IMPORTANT: The export file created by 'notebookToPython' is only intended to allow easier
# review of changes before committing code and pushing to source control. The file is not
# used within the notebook.
def notebookToPython(name, lastCell = 'top'):
    global global_lastCell

    readName = name + '.ipynb'
    writeName = name + '.py'

    if (lastCell == 'top' or lastCell == global_lastCell):
        print('Write python file')
        jupytext.write(jupytext.read(readName), writeName)
    else:
        print('Skipped write python file')

    global_lastCell = lastCell