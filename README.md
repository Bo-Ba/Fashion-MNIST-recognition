# Klasyfikacja miniatur zdjęć ubrań z wykorzystaniem konwolucyjnej sieci neuronowej.
# Wstęp
Rozwiązywane zagadnienie polega na rozpoznaniu obiektu znajdującego się na obrazku i przypisaniu mu jednej z 10 kategorii: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Każdy z ocenianych przykładów jest obrazem w skali szarości i ma wymiary 28x28. Dane są podzielone na dwie paczki: treningowa zawierająca 60 tys. obrazków i testowa zawierająca 10 tys. obrazków. 

Do wykonania zadania zdecydowałem się użyć biblioteki TensorFlow 2. Biblioteka ta posiada właściwe wszystkie narzędzia potrzebne do zrealizowania zadania, jest bardzo łatwa w użyciu oraz jej użycie jest bardzo powszechne. Co więcej jest ona dobrze zoptymalizowana oraz posiada opcję przeprowadzania obliczeń na GPU, co znacząco je przyśpiesza.
# Metodologia
Celem postawionym na początku było uzyskanie dokładności przewidywania na poziomie ok 93%. Przy pierwszym podejściu ze zwykłymi sieciami neuronowymi (Fully connected multi layer neural networks) wyniki były jednak niezadawalające. Stąd wynikła decyzja o zastosowaniu sieci konwolucyjnej. Ostateczny model jest połączeniem warstw konwolucyjnych i zwykłej sieci.

Po co warstwy konwolucyjne?
Zadaniem warstw konwolucyjnych jest nałożenie filtrów na obrazek w celu wydobycia z niego pewnych cech. Tymi cechami mogą być np. linie poziome, linie pionowe, koła itp. Wydobycie cech z obrazków pozwala sieci dopasować się znacznie lepiej. Filtry to nic innego jak małe macierze, przez które przemnażany jest obrazek. 

![filtr_przykład](https://user-images.githubusercontent.com/61791613/121871728-b8dd3480-cd04-11eb-8478-cc1c03299368.png)

*Źródło: [https://www.youtube.com/watch?v=x_VrgWTKkiM*](https://www.youtube.com/watch?v=x_VrgWTKkiM)*



Wg. literatury[^1] najbardziej odpowiednimi rozmiarami dla filtrów są 3x3 lub 5x5. Proces nakładania filtra polega na przesuwania okna po obszarze obrazu i dla każdego środkowego piksela z okna liczona jest nowa wartość. Nową wartość otrzymujemy poprzez pomnożenie pikseli z okna przez wartości z filtra i zsumowanie ich.

![filtr_gif](https://user-images.githubusercontent.com/61791613/121871785-cb576e00-cd04-11eb-87d6-d200a421460a.gif)

*Źródło: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1*

### Hiperparametry sieci konwolucyjnej

Padding – jak widać na wizualizacji powyżej nakładając filtr tracimy oryginalny wymiar obrazu. Aby tego uniknąć stosuje się sztuczne wypełnienie pikselami o wartości 0.

![padding](https://user-images.githubusercontent.com/61791613/121871832-dad6b700-cd04-11eb-92a2-f739380d6a46.gif)

*Źródło: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1*

Strides – czyli wartość o jaką filtr przeskakuje po danym obszarze.

![strides](https://user-images.githubusercontent.com/61791613/121871880-e9bd6980-cd04-11eb-92e8-1db27c8d5e2d.gif)

*Źródło: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1*

Rozmiar filtra – można go zdefiniować, jednak jak pisałem wyżej, najbardziej zaleconymi rozmiarami są 3x3 lub 5x5.

Ważnym aspektem jest to, że warstwa konwolucyjna może (a prawie zawsze tak jest) składać się z wielu filtrów. Oznacza to, że wynik na wyjściu tej warstwy będzie trójwymiarowy (w przypadku naszego zadania) i będzie wynosił 28x28xF, gdzie F – liczba użytych filtrów.

![wszystkieflitry](https://user-images.githubusercontent.com/61791613/121871920-f8a41c00-cd04-11eb-9542-01a930666d31.png)

*Źródło: https://brilliant.org/wiki/convolutional-neural-network/*

Pooling layers – Warstwy konwolucyjne często generują podobne wartości dla sąsiednich pikseli, co zwykle jest zbędną informacją. Zadaniem „poolingu” jest wybranie wartości najbardziej istotnych z danego obszaru. Zabieg ten zmniejsza nam również wielkość obrazu, co w przypadku dużej wielowymiarowości danych na wyjściu warstwy konwolucyjnej znacząco przyspiesza obliczenia. W skrócie, polega to na zmapowaniu kilku pikseli z obszaru np. 2x2 na jeden piksel. Są różne techniki, ale najczęściej używane jest wybieranie największej wartość z danego obszaru lub liczenie średniej z obszaru. 

![pooling](https://user-images.githubusercontent.com/61791613/121871961-0659a180-cd05-11eb-855d-f12d1d25af70.gif)


### Model
![model](https://user-images.githubusercontent.com/61791613/121872041-1bcecb80-cd05-11eb-92c2-fc7c94e365ae.png)

Na powyższym wycinku kodu znajduje się przygotowany przeze mnie model, który chciałbym teraz omówić.

Jest to model sekwencjny, który składa się z 6 warstw: czterech konwolacyjnych, jednej warstwy Fully Connected (FC) i warstwy wyjściowej.

Pierwsze dwie warstwy to warstwy konwolacyjne z 15 filtrami o rozmiarze 3x3. W literaturze często można przeczytać, że wymiary danych wyjściowych do warstwy powinny być takie same jak wymiary wejściowe, dlatego zdecydowałem się na użycie wcześniej opisywanego paddingu. Podczas pracy z siecią, testy potwierdzały tę teorię i kiedy rozmiary obrazków po przejściu przez warstwy się zmieniały wyniki sieci były gorsze. Inspirując się pracą: <https://arxiv.org/pdf/2001.09136v5.pdf> oraz innymi dostępnymi w Internecie zastosowałem dwie sieci konwolacyjne po sobie. Na podstawie tej samej pracy próbowałem również ograniczyć liczbę Pooling Layers do jednej, występującej tylko po drugim bloku wartsw konwolacyjncyh. Pomimo tego, że pooling niesie ze sobą sporo plusów, to sądzę, że stosując pooling zbyt często (np. po każdym bloku warstw konwolacyjnych) tracimy część danych i informacji o ocenianym obiekcie. Jednak z tego rozwiązania zrezygnowałem, co opiszę w sekcji wyniki. Po tym bloku wartst konwolacyjncyh umieszam Pooling Layer, która zmniejsza wymiary obraz z 28x28xx15 do 14x14x15.

Drugi „blok” sieci konwolacyjncyh ma już 30 filtrów ro rozmiarze 3x3. Jest to powszechnie stosowana technika, aby kolejne warstwy konwolacyjne miały więcej filtrów niż poprzednie. W swoim przypadku zdecydowałem się po prostu na podwojenie wartości poprzedniej. Umiesczona po tym bloku warstwa Pooling zmniejsza wymiary danych wyjściowych z 14x14x30 do 7x7x30.

Wyjście sieci konwolacyjnej jest wielowymiarowe, więc aby przekazać je do następnej warstwy musimy je spłaszczyć – do tego służy warstwa FLatten.

Kolejną wartswą jest wartswa FC składająca się z 735 neuronów. Jest wiele teorii na to, ile powinno być neuronów w warstwie, natomiast ja zdecydowałem się na liczbę będącą po prostu dzielnikiem poprzedniej warstwy – flatten. Warstwa ta ma być pośrednikiem pomiędzy warstwą spłaszczającą a warstwą wyjściową. Redukcja z 1470 neuronów do 10 neuronów wyjściowych byłaby po prostu z byt duża i sieć uczyłaby się dużo dłużej, i z mniejszą skutecznością.

Warto też zwrócić uwagę na „warstwy” Dropout. Jest to pierwszy zastosowany przeze mnie mechanizm zapobiegający overfittingowi. Polega on na tym, że neurony są wyłączane z pewnym, zadanym przez nas prawdopodobieństwem. Losowe wyłącznie przy każdym przejściu sprawia, że sieć nie nauczy się danych „na pamięć” zbyt szybko. 

Jako funkcji aktywacji używam ReLu, jest to chyba najpopularniejsza funkcja, która wprowadza nieliniowość do naszego modelu. Jako funkcja aktywacji na ostatniej warstwie została użyta funkcja softmax, ze względu na to, że chcę otrzymać predykcję, a więc wartości z przedziału 0 – 1.
### Dane

![dane](https://user-images.githubusercontent.com/61791613/121872083-25f0ca00-cd05-11eb-9fb6-97fa598a7941.png)

Ze względu na to, że używam warstw konwolacyjnych w których jest zawarty proces ekstrakcji cech postanowiłem nie dokonywać dużych zmian na zbiorze wejściowym. Jedyne co robię to standaryzuję dane dzieląc wartości pikseli przez 255. Operacja reshape jest niezbędna ze względu na wartwy konwolacyjne, które oczekują takiego formatu danych. Metoda to\_categorical zamienia wartości etykiet na wartości binarne przyporządkowane każdej kategorii np. dla label’a 9 dostaniemy 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1].
### Uczenie

![uczenie](https://user-images.githubusercontent.com/61791613/121872181-428d0200-cd05-11eb-91be-cfad39531d65.png)

Jako algorytmu optymalizującego użyłem Adaptive Moment Estimation (Adam). Jest on chyba najpopularniejszym teraz algorytmem optymalizacji. „Algorytm określa wartości współczynnika uczenia na podstawie średniej oraz wariancji gradientu. Następnie parametry aktualizowane są wykorzystując odpowiadające im współczynniki uczenia.”[^2]

Ze względu na to, że rozwiązywany problem jest problemem wielowymiarowej klasyfikacji, jako funkcji straty używam caterogical\_crossentropy, która dana jest wzorem:

![Obraz12](https://user-images.githubusercontent.com/61791613/121875386-aebd3500-cd08-11eb-8c85-17fd18191134.png)

Do procesu uczenia wyodrębniłem zbiór walidacyjny, który wynosi 5% zbioru treningowego. Wartość funkcji straty liczonej dla zbioru walidacyjnego po każdej epoce służy do monitorowania procesu uczenia i zapobieganiu przeuczeniu sieci. Dokładniej, wykorzystałem do tego mechanizm EarlyStopping, który zatrzyma proces uczenia w momencie, kiedy wartość funkcji straty na zbiorze walidacyjnym nie spadnie po 5 epokach uczenia. Wielkość batcha zmieniłem ze standardowych dla biblioteki TensorFlow 32 dna 64. Podczas procesu uczenia warstwy konwolacyjne dopasowują wartości filtrów, tak aby optumalizować działanie sieci. 
# Wyniki
Każdy model testowałem pięciokrotnie a następnie wyciągnąłem średnią z uzyskanych wyników.

M1 – model przedstawiony w sekcji „Metodologia”.

M2 – model z sekcji „Metodologia” bez Pooling Layer po pierwszym bloku warstw konwolacyjncyh. 

M3 - model z sekcji „Metodologia” bez warstwy FC o rozmiarze 735 przed warstwą wyjścia.

Wartości podane w tabeli są wartościami średnimi z 5 prób.

|Model|Liczba epok|Czas jednej epoki (sek)|Funkcja straty|Dokładność|
| :-: | :-: | :-: | :-: | :-: |
|M1|≈17|≈21|≈0,2007|≈93,14%|
|M2|≈13|≈34|≈0,2421|≈93,32%|
|M3|≈34|≈17|≈0,2092|≈92,64%|

Pomimo tego, że model M2 osiągnął minimalnie lepszy wynik trzeba zauważyć, że jego średnia funkcja straty jest większa o ok 20%. Interpretuję to w ten sposób, że model M2 jest bardziej niestabilny od modelu M1. Tę tezę zdawałyby się potwierdzać maksymalne i minimalne wyniki jakie osiągnęły te dwa modele. 

|Model|Min F Straty|Max F Straty|Min Dokładności|Max Dokładności|
| :-: | :-: | :-: | :-: | :-: |
|M1|0,1914|0,2157|92,94%|93,26%|
|M2|0,2123|0,2734|92,84%|93,54%|

Jak można zaobserwować „rozstrzał” na modelu bez dodatkowego Poolingu jest większy. 

Na ostateczny wybór modelu miał wpływ czas uczenia jednej epoki, dokładność osiągnięta w próbach oraz stabilność.   

W zestawieniu załączonym do zadania[^3] nie ma metody odpowiadającej mojej, jednak model zaproponowany przeze mnie osiąga lepsze wyniki od każdej z przedstawionych. 


# Użycie
Aby uruchomić model wystarczy uruchomić plik NeuralNetwork.py. Dane są zaciągnę z biblioteki TensorFlow, więc użytkownik nie musi nic importować. Trzeba również zainstalować bibliotekę TensorFlow – wystarczy użyć komendy pip install tensorflow. W przypadku jeśli chcemy użyć akceleracji z GPU’u niezbędne będzie zainstalowanie CUDA Toolkit (w przypadku karty NVidia) oraz NVIDIA cuDNN.

[^1]: https://cs231n.github.io/convolutional-networks/
[^2]: „Analiza sceny przy użyciu głębokich sieci neuronowych typu yolo” Mateusz MIKOŁAJCZYK, Arkadiusz KWASIGROCH, Michał GROCHOWSKI 
[^3]: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#
