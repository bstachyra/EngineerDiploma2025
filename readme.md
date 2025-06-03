# Praca dyplomowa inżynierska
Projekt i realizacja programu do rozpoznawania w czasie rzeczywistym gestów polskiego języka migowego. \
Autor: Bartosz Stachyra
## Instrukcja uruchomienia
Poniższa instrukcja zakłada posiadanie Python w wersji 3.11 na stacji użytkownika, z kompatybilną wersją pypi.
### Linux
Stwórz środowisko wirtualne:
```
python3 -m venv venv
```
Aktywuj środowisko wirtualne:
```
source venv/bin/activate
```
Zainstaluj zależności:
```
pip install -r requirements.txt
```
Uruchom aplikację:
```
python main_application/app.py
```
Po zakończeniu działania aplikacji deaktywuj środowisko wirtualne:
```
deactivate
```
### Windows
Stwórz środowisko wirtualne:
```
python3 -m venv venv
```
Aktywuj środowisko wirtualne:
```
.\venv\Scripts\activate
```
Zainstaluj zależności:
```
pip install -r requirements.txt
```
Uruchom aplikację:
```
python main_application\app.py
```
Po zakończeniu działania aplikacji deaktywuj środowisko wirtualne:
```
deactivate
```