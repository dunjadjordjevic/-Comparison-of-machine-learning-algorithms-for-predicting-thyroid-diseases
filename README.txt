Ovaj projekat predstavlja implementaciju istrazivackog dela koji je radjen u sklopu master rada koji ima za cilj da implementira sistem koji ce
da omoguci implementaciju cetiri modela masinskog ucenja (KNN, Stabla odlucivanja, Slucajne sume, Naivni Bajesov klasifikator) u cilju omogucavanja
predikcije dijagnoze pacijenata, nad nepoznatim skupovima podataka.

Koriscen je programski jezik Python. Aplikacija je implementirana u istom programskom jeziku, koriscen je framework Django. Korisceni IDE - PyCharm.

Struktura projekta je sledeca:
- application_screenshots - direktorijum gde se mogu naci slike ekrana kreirane internet aplikacije Thyroid disease prediction app
- static - direktorijum sa slikama koriscenim u aplikaciji, kao i css folderom za stilizovanje aplikacije
- thyroid_project - direktorijum u kojem se nalazi implementacija cetiri pomenuta modela masinskog ucenja (from scratch, nalaze se u models/algorithms)
                    kao i evaluatora - fajl koji omogucava evaluaciju za sve modele masinskog ucenja, sto podrazumeva implementaciju evaluate
                    metode; kada je potrebno da se izvrsi evaluacija odredjenog modela masinskog ucenja, u ovoj metodi potrebno je odkomentarisati
                    odgovarajuci deo koda, i u glavnoj metodi postaviti flag za evaluaciju na True (vazi samo kada se pokrece manuelno projekat,
                    bez internet aplikacije, radi provere);
                    ovaj direktorijum sadrzi takodje i resources poddirektorijum sa generisanim datasetovima, koji ce se koristiti u internet aplikaciji
                    za predikciju nad nepoznatim skupovima podataka, a koji su nastali nasumicnim izborom podataka iz obradjenog radnog skupa; takodje sadrzi
                    generisane slike koje predstavljaju medjurezultate procesa pretprocesiranja podataka, kao i generated_prediction_datasets
                    folder u kome se cuva rezultat predikcije nepoznatog ucitanog skupa podataka
                    thyroidDF.csv predstavlja fajl u kome se nalazi sacuvan Kaggle-ov skup podataka za obradu u izvornom obliku sa oko 9000 zapisa i 31 kolonom
- thyroid_webapp - direktorijum koji ima default-nu strukturu jedne Django aplikacije i u kojem se nalazi poddirektorijum templates u kome
                   je implementiran FE deo ove aplikacije;
                   glavna logika i request koji se salje sa FE na BE kako bi se povezale ove dve celine projekta i omogucila nalazis se u
                   views.py fajlu

Pokretanje aplikacije omoguceno je pokretanjem komande iz Git Basha: python manage.py runserver, nakon sto se pozicionira u folder thyroid_webapp odakle je manage.py
fajl vidljiv. Takodje je moguce pokretati projekat iz PyCharma, pod pretpostavkom da je podeseno Anaconda virtualno okruzenje sa svim instaliranim paketima.
Korisceni paketi su vidljivi u requirements.txt fajlu i mogu se skinuti koriscenjem ovog fajla po potrebi.