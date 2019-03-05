#! /bin/bash
rm tesi.aux tesi.blg tesi.out tesi.pdf tesi.toc tesi.log
pdflatex tesi
bibtex tesi
pdflatex tesi
pdflatex tesi