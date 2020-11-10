### 2020 Presidential Election Voting Data
Run `get_data.py` to scrape all voting data from the New York Times (while the api lasts)  
Then edit and run `analyze.py` to generate some plots.

A nice utility for combining all png's into one pdf is `img2pdf`:
```
sudo apt-get install img2pdf
img2pdf *.png -o all_pngs.pdf
```

If using WSL, need an Xserver to plot without exceptions raised, but it is not needed to display. 
Just run `wslview all_pngs.pdf` 

