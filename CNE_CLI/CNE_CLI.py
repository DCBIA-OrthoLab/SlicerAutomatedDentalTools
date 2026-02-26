#!/usr/bin/env python-real

import sys, argparse ,shutil, os, time
print("CNE_CLI.py run")

def main(args):

    # Récupération des arguments
    notesFolder_input = args.notesFolder_input
    modelType = args.modelType
    notesType = args.notesType
    notesFolder_output = args.notesFolder_output

    # strat 
    print("<filter-start><filter-name>Extraction des notes</filter-name></filter-start>", flush=True)
    
    # ---------------------------------------------------------
    # 2. ÉTAPE 1 : Lecture (Progression 10%)
    # ---------------------------------------------------------
    print("<filter-progress>0.10</filter-progress>", flush=True)
    print("<filter-comment>Lecture du dossier d'entrée...</filter-comment>", flush=True)
    


    time.sleep(2)



    # ---------------------------------------------------------
    # 5. FIN : La barre passe à 100% et se ferme
    # ---------------------------------------------------------
    print("<filter-progress>1.00</filter-progress>", flush=True)
    print("<filter-comment>Terminé !</filter-comment>", flush=True)


    print("<filter-end><filter-name>Extraction des notes</filter-name></filter-end>", flush=True)
    print("\nCNE_CLI.py done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('notesFolder_input',type=str)
    parser.add_argument('modelType',type=str)
    parser.add_argument("notesType",type=str)
    parser.add_argument('notesFolder_output',type=str)

    args = parser.parse_args()
    main(args)
