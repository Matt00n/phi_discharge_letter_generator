"""Demographically-grounded German name pools.

Diversity is a primary project goal, so names are sampled from origin-weighted
pools reflecting the population actually seen in German clinics - not just
majority-German names. This directly attacks the memorization shortcut: an
anonymizer cannot win by learning a tiny closed vocabulary.

Weights are approximate and intended to be representative rather than exact.
Extend the pools freely; the sampler does not assume any fixed size.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# origin -> sampling weight (calibrated to German hospital inpatient demographics,
# reflecting immigration statistics from Statistisches Bundesamt / Mikrozensus 2022).
# Hospital populations skew slightly more toward migration backgrounds than the
# general population (~26% vs ~23%) due to age structure and socioeconomic factors.
ORIGIN_WEIGHTS: dict[str, float] = {
    "de": 0.60,
    "tr": 0.10,
    "ar": 0.07,
    "pl": 0.06,
    "ru": 0.05,
    "ex_yu": 0.04,
    "it": 0.03,
    "gr": 0.02,
    "vn": 0.01,
    "other_eu": 0.02,
}

# fmt: off
GIVEN_NAMES_M: dict[str, list[str]] = {
    "de": [
        # Modern popular
        "Lukas", "Felix", "Jonas", "Maximilian", "Paul", "Elias", "Hans", "Karl",
        "Werner", "Helmut", "Dieter", "Matthias", "Sebastian", "Tobias", "Friedrich",
        "Jan", "Thomas", "Michael", "Stefan", "Andreas", "Martin", "Peter",
        "Alexander", "Christian", "Daniel", "Florian", "Markus", "Bernd",
        "Gerhard", "Jürgen", "Wolfgang", "Heinrich", "Uwe", "Rainer", "Manfred",
        "Klaus", "Günter", "Norbert", "Roland", "Ralf", "Frank", "Oliver",
        "Philipp", "Tim", "Benjamin", "Niklas", "Leon", "David", "Simon", "Moritz",
        # Older / regional
        "Siegfried", "Wilfried", "Hartmut", "Eckhard", "Horst", "Lothar",
        "Detlef", "Torsten", "Holger", "Jörg", "Volker", "Reinhard", "Gerd",
        "Karsten", "Sören", "Malte", "Arne", "Henning", "Thilo", "Clemens",
        "Benedikt", "Konrad", "Albrecht", "Ruprecht", "Willibald", "Ottmar",
        "Balthasar", "Korbinian", "Xaver", "Leander", "Laurenz", "Tristan",
        "Thorben", "Jannik", "Marlon", "Cedric", "Lennart", "Fynn", "Tilo",
        "Ansgar", "Ludger", "Folkert", "Hauke", "Fiete", "Lasse", "Nils",
        "Wulf", "Armin", "Hannes", "Jochen", "Heiko",
        # Additional rare / old-fashioned / dialect
        "Adalbert", "Alois", "Arnulf", "Bernhard", "Berthold", "Bodo", "Bruno",
        "Burkhard", "Cord", "Diethelm", "Eberhard", "Egon", "Engelbert", "Erwin",
        "Eugen", "Ferdinand", "Gebhard", "Georg", "Gottfried", "Gottlieb",
        "Gregor", "Günther", "Gustav", "Harald", "Hasso", "Heribert", "Hermann",
        "Hilmar", "Hubertus", "Ingo", "Jakob", "Joachim", "Josef", "Jost",
        "Justus", "Lambert", "Leopold", "Lorenz", "Ludwig", "Meinhard",
        "Nepomuk", "Nikolaus", "Ortwin", "Oskar", "Otfried", "Otto", "Pirmin",
        "Quirin", "Raimund", "Reinhold", "Richard", "Robert", "Rupert", "Severin",
        "Sigmund", "Silvester", "Theodor", "Tilman", "Udo", "Ulrich", "Valentin",
        "Viktor", "Vinzenz", "Wendelin", "Wigbert", "Willhelm", "Wolfram",
        # Young modern
        "Aaron", "Adrian", "Anton", "Benno", "Bjarne", "Carl", "Constantin",
        "Dominik", "Emilian", "Fabian", "Gabriel", "Henrik", "Ian", "Julius",
        "Kilian", "Liam", "Marian", "Nico", "Oskar", "Pascal", "Quentin",
        "Rafael", "Samuel", "Theodor", "Vincent", "Yannik",
    ],
    "tr": [
        "Mehmet", "Mustafa", "Ahmet", "Emre", "Burak", "Yusuf", "Hasan", "Ali",
        "Murat", "Serkan", "Kemal", "Cem", "Erkan", "Osman", "Selim", "İbrahim",
        "Ömer", "Halil", "Orhan", "Bayram", "Cengiz", "Deniz", "Eren", "Arda",
        "Kaan", "Baran", "Furkan", "Enes", "Volkan", "Sinan",
        "Gökhan", "Tolga", "Onur", "Umut", "Alper", "Taner", "Kadir", "Recep",
        "Tarık", "Sedat", "Engin", "Oktay", "Ferhat", "Bilge", "Berkay",
        "Yiğit", "Adem", "Fatih", "Tugay", "Barış", "Doruk", "Atakan",
        "Emir", "Batuhan", "Oğuz", "Polat", "Koray", "Sefa", "Mert", "Alparslan",
        # Additional
        "Süleyman", "Tuncay", "Necati", "Hikmet", "Metin", "Fikret", "Sezgin",
        "Nurettin", "Bülent", "Levent", "Çağdaş", "Kenan", "Güven", "İlhan",
        "Burhan", "Nuri", "Salih", "Tayfun", "Vedat", "Yakup", "Zeki",
        "Harun", "Muhammed", "Hamit", "Nihat", "Remzi", "Rıza", "Şükrü",
        "Tuğrul", "Ümit", "Veli", "Yılmaz", "Zafer", "Abdullah", "Bahadır",
        "Celal", "Doğukan", "Ekrem", "Feridun", "Gürsel", "Hüseyin",
        "İsmet", "Kürşat", "Mahir", "Necdet", "Oğuzhan", "Ramazan",
        "Sadettin", "Teoman", "Uğur", "Yavuz", "Zülfikar",
    ],
    "ar": [
        "Mohammed", "Omar", "Khaled", "Tarek", "Samir", "Bilal", "Yasin", "Karim",
        "Ahmed", "Hassan", "Youssef", "Fadi", "Nabil", "Walid", "Rami", "Adel",
        "Hamza", "Amin", "Sami", "Rachid", "Amir", "Ziad", "Bassem", "Mazen",
        "Elias", "Imad", "Jamal", "Mustafa", "Hadi", "Ibrahim",
        "Abdelkarim", "Abderrahim", "Mounir", "Fouad", "Anouar", "Soufian",
        "Zakaria", "Oussama", "Mehdi", "Ilyas", "Ayoub", "Badr", "Chafik",
        "Driss", "Farid", "Hakim", "Ismail", "Jawad", "Lahcen", "Nadir",
        "Othman", "Qasim", "Redouane", "Sofiane", "Tawfiq", "Wael",
        "Yassir", "Nizar", "Hicham", "Lotfi",
        # Additional
        "Abdel", "Anwar", "Ashraf", "Ayman", "Bahaa", "Chakib", "Djamel",
        "Essam", "Gamal", "Hafid", "Ihab", "Issam", "Kamal", "Lakhdar",
        "Majid", "Murad", "Nader", "Osama", "Rafik", "Saeed", "Salim",
        "Slim", "Talal", "Tahar", "Yasser", "Younes", "Abdelhak", "Brahim",
        "Cherif", "Faysal", "Hichem", "Kais", "Mansour", "Noureddine",
        "Ridha", "Saad", "Sadek", "Seddik", "Wahid", "Yacine",
        "Abdelaziz", "Abdellatif", "Abdelmalek", "Benali", "Djaafar",
        "Habib", "Hamid", "Mokhtar", "Mourad", "Nacer", "Rabah",
        "Rashid", "Slimane", "Toufik", "Zouhair",
    ],
    "pl": [
        "Krzysztof", "Tomasz", "Andrzej", "Piotr", "Marcin", "Grzegorz", "Paweł",
        "Jan", "Adam", "Marek", "Michał", "Jakub", "Łukasz", "Rafał", "Dariusz",
        "Robert", "Kamil", "Wojciech", "Sławomir", "Janusz", "Zbigniew",
        "Stanisław", "Artur", "Damian", "Bartosz", "Mateusz", "Maciej",
        "Przemysław", "Dominik", "Sebastian",
        "Bogdan", "Czesław", "Eugeniusz", "Feliks", "Henryk", "Ireneusz",
        "Jerzy", "Kazimierz", "Lech", "Mirosław", "Norbert", "Oskar",
        "Radosław", "Ryszard", "Tadeusz", "Wiesław", "Zdzisław", "Teodor",
        "Wiktor", "Borys", "Cezary", "Hubert", "Igor", "Leon", "Nikodem",
        "Olgierd", "Patryk", "Szymon", "Tymon", "Franciszek",
        # Additional
        "Aleksander", "Antoni", "Bartłomiej", "Benedykt", "Bronisław",
        "Cyprian", "Edmund", "Emanuel", "Filip", "Florian", "Gerard",
        "Gustaw", "Hieronim", "Ignacy", "Jacek", "Julian", "Kornel",
        "Krystian", "Leszek", "Ludwik", "Maksymilian", "Mariusz",
        "Mieczysław", "Napoleon", "Natan", "Oktawian", "Oliwier",
        "Pankracy", "Rajmund", "Roman", "Seweryn", "Sylwester",
        "Tobiasz", "Urban", "Waldemar", "Witold", "Władysław",
        "Zenon", "Zygmunt", "Błażej", "Dorian",
    ],
    "ru": [
        "Wladimir", "Sergej", "Dmitri", "Igor", "Alexej", "Andrej", "Nikolai",
        "Viktor", "Juri", "Maxim", "Pawel", "Oleg", "Boris", "Konstantin",
        "Wadim", "Anatoli", "Gennadi", "Leonid", "Jewgeni", "Fjodor",
        "Arkadi", "Timur", "Ilja", "Roman", "Denis", "Artjom", "Witalij",
        "Ruslan", "Valeri", "Michail",
        "Stanislaw", "Wenjamin", "Wsewolod", "Gleb", "Kirill", "Semjon",
        "Georgi", "Pjotr", "Lew", "Gordej", "Nikita", "Arsen", "Wladislaw",
        "Jaroslaw", "Bogdan", "Daniil", "Matwej", "Stepan", "Platon",
        "Grigori", "Sachar", "Filipp", "Innokenti", "Prokop", "Tichon",
        "Wladlen", "Eduard", "Albert", "Artur", "Robert",
        # Additional
        "Afanassi", "Alexander", "Arsenij", "Borissow", "Dawid", "Demjan",
        "Foma", "Gavriil", "Herman", "Iwan", "Jakow", "Jefim", "Jemeljan",
        "Kondrati", "Kusma", "Lasar", "Lukjan", "Makar", "Mark", "Miron",
        "Nazar", "Nestor", "Ossip", "Pantelejmon", "Rodion", "Saweli",
        "Taras", "Timofej", "Trofim", "Wlas", "Zakhar", "Serafim",
        "Apollon", "Demid", "Jermolaj", "Kapiton", "Lukian", "Mitrofan",
        "Onufri", "Pachom", "Potap", "Sawwa", "Seliman", "Spartак",
        "Tichon", "Weniamin", "Wsewolod", "Apollinari", "Kliment",
        "Methodij", "Parfjon", "Silas",
    ],
    "ex_yu": [
        "Dragan", "Goran", "Milan", "Ivan", "Stjepan", "Nikola", "Marko",
        "Dejan", "Zoran", "Branko", "Darko", "Mladen", "Tomislav", "Nenad",
        "Slobodan", "Predrag", "Saša", "Boris", "Damir", "Luka", "Mate",
        "Željko", "Miroslav", "Dušan", "Aleksandar", "Stevo", "Vedran",
        "Hrvoje", "Filip", "Alen",
        "Ljubomir", "Radovan", "Vojislav", "Blagoje", "Čedomir", "Dragomir",
        "Milivoj", "Nebojša", "Ognjen", "Perica", "Ratko", "Srđan",
        "Vlatko", "Zdenko", "Semir", "Edin", "Admir", "Haris", "Nermin",
        "Zlatan", "Milorad", "Rajko", "Slavko", "Velimir", "Đorđe",
        "Radmilo", "Miodrag", "Tihomir", "Bratislav", "Miloš",
        # Additional
        "Andrija", "Bogoljub", "Časlav", "Danilo", "Dragutin", "Dženan",
        "Fadil", "Gojko", "Hamza", "Igor", "Jasmin", "Kenan", "Lazar",
        "Mirsad", "Muhamed", "Nedim", "Omer", "Pavle", "Ranko", "Simo",
        "Siniša", "Todor", "Uroš", "Vasilije", "Vojo", "Zvonimir",
        "Budimir", "Dimitrije", "Jovan", "Kosta", "Milenko", "Momčilo",
        "Obrad", "Petar", "Radoš", "Spasoje", "Stanimir", "Svetozar",
        "Toma", "Vitomir",
    ],
    "it": [
        "Giuseppe", "Marco", "Antonio", "Luca", "Francesco", "Salvatore",
        "Giovanni", "Roberto", "Andrea", "Alessandro", "Matteo", "Davide",
        "Vincenzo", "Stefano", "Domenico", "Massimo", "Fabio", "Pietro",
        "Nicola", "Simone", "Lorenzo", "Carlo", "Emanuele", "Federico",
        "Gianluca", "Paolo", "Bruno", "Claudio", "Enrico", "Riccardo",
        "Pasquale", "Gennaro", "Carmine", "Raffaele", "Gaetano", "Agostino",
        "Cesare", "Dario", "Enzo", "Filippo", "Giacomo", "Ignazio",
        "Ludovico", "Mirko", "Nino", "Osvaldo", "Renato", "Silvio",
        "Umberto", "Vittorio", "Achille", "Baldassare", "Corrado",
        "Erminio", "Fulvio", "Guglielmo", "Leone", "Marcello", "Oreste", "Rocco",
        # Additional
        "Aldo", "Alfredo", "Amedeo", "Angelo", "Arturo", "Augusto",
        "Benito", "Camillo", "Costantino", "Daniele", "Diego", "Donato",
        "Edoardo", "Ernesto", "Ettore", "Fausto", "Fernando", "Gerardo",
        "Giancarlo", "Gino", "Giorgio", "Giuliano", "Gregorio", "Guido",
        "Italo", "Ivano", "Lauro", "Lino", "Livio", "Luciano",
        "Marino", "Mauro", "Natale", "Nunzio", "Ottavio", "Pier",
        "Primo", "Remo", "Rolando", "Romeo", "Ruggero", "Sandro",
        "Sergio", "Tiziano", "Tullio", "Ugo", "Valerio", "Vito",
    ],
    "gr": [
        "Dimitrios", "Georgios", "Nikolaos", "Konstantinos", "Christos",
        "Panagiotis", "Vasileios", "Ioannis", "Spyridon", "Athanasios",
        "Evangelos", "Alexandros", "Theodoros", "Stavros", "Michail",
        "Andreas", "Petros", "Ilias", "Fotios", "Kostas", "Dimitris",
        "Apostolos", "Sotirios", "Charalampos", "Emmanouil",
        "Anastasios", "Dionysios", "Efstathios", "Grigorios", "Herakles",
        "Kyriakos", "Lambros", "Miltiadis", "Odysseas", "Pantelis",
        "Serafeim", "Themistoklis", "Triantafyllos", "Vasilios",
        "Xenofon", "Achilleas", "Aristeidis", "Chrysanthos", "Evaggelos",
        "Filippos", "Gerasimos", "Harilaos", "Leonidas", "Nektarios",
        "Polykarpos",
        # Additional
        "Aggelos", "Antonios", "Charalabos", "Demetrios", "Efthymios",
        "Eustathios", "Fotis", "Haralambos", "Iosif", "Kallinikos",
        "Konstantinos", "Lazaros", "Loukas", "Makarios", "Neofytos",
        "Nikiforos", "Orestis", "Pavlos", "Prokopios", "Rafail",
        "Savvas", "Socrates", "Stefanos", "Taxiarchis", "Thanasis",
        "Timotheos", "Vasilis", "Vlasios", "Yerasimos", "Zacharias",
        "Avraam", "Damaskinos", "Epameinondas", "Fokion", "Gavriil",
        "Herodotos", "Iason", "Kleanthis", "Lefteris", "Manolis",
        "Nestor", "Panayotis", "Prodromos", "Sokratis", "Symeon",
        "Theofilos", "Tryfon", "Xenophon", "Aristotelis", "Periklis",
    ],
    "vn": [
        "Minh", "Quang", "Tuan", "Hoang", "Duc", "Nam", "Long", "Hieu",
        "Thanh", "Phong", "Bao", "Khoa", "Trung", "Dat", "Huy", "Tien",
        "An", "Khanh", "Duy", "Son", "Hung", "Cuong", "Tai", "Binh", "Vinh",
        "Lam", "Hai", "Quan", "Thinh", "Loc", "Tam", "Thang", "Vu",
        "Nhat", "Tri", "Phuc", "Hao", "Sang", "Nghia", "Tung",
        "Phat", "Dung", "Truong", "Khang", "Quoc", "Toan", "Linh",
        "Danh", "Kien", "Loi",
        # Additional
        "Bach", "Can", "Chien", "Dan", "Diep", "Giang", "Ha", "Hien",
        "Hoa", "Huan", "Huu", "Ky", "Luan", "Man", "Nang", "Nghi",
        "Nhan", "Phu", "Quyen", "Si", "Tan", "Thach", "Thien", "Thiep",
        "Tho", "Thong", "Tien", "Tin", "Toan", "Trong", "Truc",
        "Tuan", "Tuyen", "Van", "Viet", "Vy", "Xuan", "Yen",
        "Dinh", "Luc", "Ngu", "Phap", "Quang", "Tai", "Uy", "Vuong",
    ],
    "other_eu": [
        "João", "Pedro", "Carlos", "Henrik", "Lars", "András", "Miguel",
        "Fernando", "Rafael", "Tiago", "Erik", "Gustav", "Oscar", "Matias",
        "Sven", "Nils", "Kristian", "Balázs", "Gábor", "Tamás", "Attila",
        "Raul", "Adrián", "Tomáš", "Jakub", "Ondřej", "Petar", "Emil",
        "Ion", "Florin",
        "Diogo", "Gonçalo", "Rúben", "Nuno", "Álvaro", "Leif", "Björn",
        "Stellan", "Torbjörn", "Ulf", "Ákos", "Bence", "Dániel", "Levente",
        "Zoltán", "Dragoș", "Marius", "Andrei", "Cosmin", "Vlad",
        "Miroslav", "Branislav", "Vojtěch", "Aleš", "Dušan", "Milán",
        "Nikolaj", "Bogomil", "Krasimir", "Zdravko",
        # Additional
        "Afonso", "Bruno", "Duarte", "Henrique", "Lourenço", "Salvador",
        "Vasco", "Axel", "Eskil", "Gunnar", "Halvard", "Ingvar", "Knut",
        "Ragnar", "Sigurd", "Viggo", "Csaba", "Elemér", "Géza", "Imre",
        "Jenő", "Kálmán", "Lajos", "Miklós", "Sándor", "Tibor",
        "Vasile", "Gheorghe", "Mihai", "Radu", "Sorin", "Tudor",
        "Bohuslav", "Dalibor", "Jaromír", "Ladislav", "Mojmír", "Přemysl",
        "Rostislav", "Stanislav", "Václav", "Vlastimil", "Zdeněk",
        "Dimitar", "Hristo", "Lyubomir", "Simeon", "Todor", "Yordan",
    ],
}

GIVEN_NAMES_F: dict[str, list[str]] = {
    "de": [
        # Common modern
        "Anna", "Marie", "Sophia", "Emma", "Hannah", "Lena", "Ursula", "Helga",
        "Gisela", "Ingrid", "Renate", "Katharina", "Johanna", "Charlotte", "Erika",
        "Monika", "Brigitte", "Elisabeth", "Gertrud", "Hildegard", "Irmgard",
        "Karin", "Sabine", "Petra", "Claudia", "Andrea", "Birgit", "Susanne",
        "Christina", "Sandra", "Stefanie", "Julia", "Martina", "Heike",
        "Laura", "Lisa", "Lea", "Mia", "Emily", "Clara", "Amelie", "Luisa",
        "Franziska", "Maria", "Barbara", "Margarete", "Theresia", "Dorothea",
        "Eva", "Bettina",
        # Older / regional
        "Rosemarie", "Lieselotte", "Elfriede", "Waltraud", "Anneliese",
        "Hannelore", "Roswitha", "Mechthild", "Adelheid", "Brunhilde",
        "Edeltraud", "Almut", "Siglinde", "Ortrud", "Dagmar", "Ilse",
        "Hedwig", "Thekla", "Frieda", "Wilhelmine", "Ottilie", "Kunigunde",
        "Gerlinde", "Christa", "Ingeborg", "Annemarie", "Marianne", "Ruth",
        "Gudrun", "Herta", "Traute", "Wiltrud", "Rotraut", "Bärbel",
        "Heidi", "Frauke", "Wiebke", "Imke", "Silke", "Birte", "Maike",
        "Svenja", "Janina", "Nele", "Antje", "Jette", "Friederike",
        "Annika", "Carolin", "Vivien",
        # Additional rare / old-fashioned / dialect
        "Adele", "Agathe", "Agnes", "Albertine", "Amalia", "Auguste",
        "Bertha", "Cäcilia", "Constanze", "Cordula", "Edith", "Eleonore",
        "Ernestine", "Felicitas", "Flora", "Gabriele", "Genoveva",
        "Gertrude", "Gisbert", "Gretel", "Hanne", "Henriette", "Hermine",
        "Ida", "Irene", "Irmtraud", "Isolde", "Josefine", "Juliane",
        "Karoline", "Klara", "Leonore", "Liesel", "Lisbeth", "Lore",
        "Lucie", "Magda", "Margot", "Marlene", "Mathilde", "Natalie",
        "Paula", "Resi", "Rosa", "Rosel", "Sigrid", "Sophie",
        "Therese", "Trude", "Ulla", "Veronika", "Wanda",
        # Young modern
        "Amelie", "Celine", "Elena", "Emilia", "Finja", "Greta",
        "Helena", "Ida", "Johanna", "Klara", "Lara", "Leonie",
        "Lilly", "Lina", "Maja", "Marlene", "Mathilda", "Mila",
        "Nora", "Paulina", "Romy", "Sara", "Stella", "Tessa",
        "Valentina", "Zoe",
    ],
    "tr": [
        "Ayşe", "Fatma", "Emine", "Zeynep", "Elif", "Hatice", "Merve", "Sevim",
        "Gülcan", "Derya", "Özlem", "Leyla", "Esra", "Nur", "Sibel", "Dilek",
        "Hülya", "Seda", "Filiz", "Nihal", "Pınar", "Melek", "Büşra", "Cansu",
        "Ebru", "Ayla", "Gamze", "Asli", "Ceren", "İrem",
        "Gülşen", "Sevinç", "Naciye", "Havva", "Fadime", "Gülay", "Gönül",
        "Nurgül", "Songül", "Aysel", "Birsen", "Münevver", "Şerife",
        "Tülay", "Yıldız", "Bahar", "Duygu", "Ece", "Gizem", "Hazal",
        "İpek", "Lale", "Nesrin", "Perihan", "Sevgi", "Tuba", "Ülkü",
        "Beyza", "Selin", "Simge",
        # Additional
        "Arzu", "Berna", "Cemile", "Defne", "Eylül", "Feraye", "Gül",
        "Hacer", "İlknur", "Kadriye", "Kezban", "Mediha", "Nalan",
        "Neriman", "Nilüfer", "Nuray", "Oya", "Pembe", "Rabia", "Reyhan",
        "Rukiye", "Sabiha", "Saadet", "Selma", "Serpil", "Şeyma",
        "Tuğba", "Vildan", "Yasemin", "Zuhal", "Zübeyde",
        "Behiye", "Cahide", "Dilara", "Elif", "Figen", "Gonca",
        "Handan", "Hürriyet", "İnci", "Jale", "Mehtap", "Müjgan",
        "Nazan", "Nevin", "Nurten", "Pakize", "Raziye", "Sanem",
        "Süheyla", "Türkan",
    ],
    "ar": [
        "Fatima", "Aisha", "Layla", "Nour", "Mariam", "Yasmin", "Salma", "Huda",
        "Samira", "Amina", "Rania", "Dina", "Sara", "Lina", "Jana", "Hana",
        "Mona", "Rana", "Leila", "Souad", "Nawal", "Wafa", "Sabah", "Ghada",
        "Hayat", "Malak", "Noura", "Dalal", "Inaam", "Bushra",
        "Abir", "Chaima", "Dounia", "Fatiha", "Habiba", "Ikram", "Jamila",
        "Khadija", "Latifa", "Meriem", "Nabila", "Oumaima", "Rajaa",
        "Siham", "Thouraya", "Widad", "Zahira", "Btissam", "Hanane",
        "Loubna", "Safaa", "Soumia", "Aicha", "Bouchra", "Fadoua",
        "Hafsa", "Jihane", "Mouna", "Rim", "Zineb",
        # Additional
        "Asma", "Bahia", "Basma", "Dalila", "Farida", "Fouzia", "Ghizlane",
        "Halima", "Houda", "Karima", "Lamia", "Leïla", "Malika", "Meryem",
        "Nadia", "Najat", "Naima", "Rachida", "Rabiaa", "Saida",
        "Sakina", "Sanaa", "Sirine", "Soukaina", "Soraya", "Touria",
        "Yasmina", "Zakia", "Zohra", "Zubaida",
        "Ahlam", "Badria", "Djamila", "Faiza", "Hayfa", "Ilham",
        "Inas", "Lubna", "Madiha", "Maryam", "Maysoon", "Munira",
        "Nabiha", "Raghda", "Rasha", "Rawda", "Sahar", "Sawsan",
        "Suha", "Thuraya",
    ],
    "pl": [
        "Katarzyna", "Agnieszka", "Małgorzata", "Anna", "Magdalena", "Ewa",
        "Joanna", "Monika", "Beata", "Dorota", "Iwona", "Barbara", "Renata",
        "Elżbieta", "Justyna", "Marta", "Aleksandra", "Natalia", "Karolina",
        "Weronika", "Zofia", "Patrycja", "Sylwia", "Kinga", "Paulina",
        "Dominika", "Aneta", "Bożena", "Jolanta", "Teresa",
        "Celina", "Danuta", "Felicja", "Grażyna", "Halina", "Izabela",
        "Jadwiga", "Krystyna", "Lidia", "Mirosława", "Otylia", "Regina",
        "Stanisława", "Urszula", "Wanda", "Zenobia", "Blanka", "Czesława",
        "Genowefa", "Helena", "Lucyna", "Mariola", "Nikola", "Olga",
        "Sabina", "Tatiana", "Violetta", "Zuzanna", "Antonina", "Bogusława",
        # Additional
        "Aldona", "Bernadeta", "Bronisława", "Celestyna", "Dagmara",
        "Edyta", "Eugenia", "Filipina", "Gabriela", "Hanna", "Ilona",
        "Irena", "Janina", "Kamila", "Klaudia", "Kornelia", "Larysa",
        "Leokadia", "Ludmiła", "Łucja", "Malwina", "Milena", "Nadia",
        "Oliwia", "Rozalia", "Stefania", "Tekla", "Waleria",
        "Wioleta", "Żaneta",
    ],
    "ru": [
        "Olga", "Natascha", "Irina", "Swetlana", "Tatjana", "Ludmila", "Galina",
        "Jelena", "Valentina", "Nadeschda", "Maria", "Tamara", "Larissa",
        "Anastasia", "Jekaterina", "Julia", "Oxana", "Wera", "Nina",
        "Alexandra", "Polina", "Darja", "Sofija", "Kristina", "Marina",
        "Raissa", "Sinaida", "Ljubow", "Alla", "Inna",
        "Antonina", "Walentina", "Klawdija", "Praskowja", "Anfisa",
        "Zinaida", "Rimma", "Nelli", "Elwira", "Rosa", "Emma", "Ida",
        "Frieda", "Lydia", "Margareta", "Natalia", "Weronika", "Switlana",
        "Milena", "Jana", "Alina", "Dina", "Kira", "Lara", "Renata",
        "Xenia", "Warwara", "Agnessa", "Ewgenija", "Margarita",
        # Additional
        "Agrafena", "Awdotja", "Domna", "Feodora", "Glafira", "Jekaterina",
        "Kapitolina", "Lukerja", "Matrjona", "Melanija", "Nadescha",
        "Olimpiada", "Pelageja", "Praskowja", "Serafima", "Ustinja",
        "Warwara", "Wassa", "Darja", "Nadezhda", "Tatiana", "Vera",
        "Zinaida", "Sofja", "Ludmilla", "Iskra", "Maja", "Rada",
        "Sweta", "Tanja",
    ],
    "ex_yu": [
        "Jelena", "Ana", "Marija", "Ivana", "Snežana", "Dragana", "Mirjana",
        "Vesna", "Milica", "Gordana", "Ljiljana", "Tatjana", "Nataša",
        "Dubravka", "Jasmina", "Maja", "Nina", "Sanja", "Branka", "Slavica",
        "Zorica", "Dušanka", "Biljana", "Ankica", "Dijana", "Senka",
        "Vedrana", "Lana", "Tanja", "Lidija",
        "Radmila", "Stanislava", "Živana", "Bosiljka", "Cvetanka",
        "Danica", "Drenka", "Grozdana", "Jovanka", "Kosara", "Leposava",
        "Milojka", "Nadežda", "Olivera", "Persida", "Radojka", "Slađana",
        "Svetlana", "Tomislava", "Višnja", "Zdravka", "Almira", "Emina",
        "Lejla", "Amra", "Sabina", "Enisa", "Mirsada", "Samra", "Mediha",
        # Additional
        "Borka", "Cvijeta", "Darinka", "Dušica", "Đurđica", "Gorana",
        "Jadranka", "Kata", "Ljubica", "Melanija", "Milana", "Mira",
        "Nada", "Nevenka", "Olga", "Rada", "Ružica", "Slavojka",
        "Snježana", "Stanka", "Štefica", "Teodora", "Verica", "Vinka",
        "Zlata", "Zorana", "Adisa", "Azra", "Dženana", "Fikreta",
        "Hajrija", "Ifeta", "Jasna", "Merima", "Nasiha", "Ramiza",
        "Refika", "Sadeta", "Šehida", "Zemka",
    ],
    "it": [
        "Giulia", "Francesca", "Maria", "Chiara", "Sofia", "Valentina",
        "Alessia", "Martina", "Laura", "Sara", "Elena", "Anna", "Silvia",
        "Paola", "Rosa", "Angela", "Lucia", "Federica", "Elisa", "Roberta",
        "Giorgia", "Ilaria", "Simona", "Monica", "Claudia", "Cristina",
        "Daniela", "Serena", "Patrizia", "Antonella",
        "Concetta", "Filomena", "Genoveffa", "Immacolata", "Giuseppina",
        "Assunta", "Addolorata", "Carmela", "Grazia", "Nunzia", "Oriana",
        "Piera", "Rosalia", "Silvana", "Teresa", "Viviana", "Adelaide",
        "Beatrice", "Carlotta", "Donatella", "Emanuela", "Flavia",
        "Giovanna", "Isabella", "Loretta", "Marta", "Noemi", "Ornella",
        "Raffaella", "Sabrina",
        # Additional
        "Agata", "Alberta", "Aurora", "Bianca", "Caterina", "Cinzia",
        "Delfina", "Eleonora", "Eugenia", "Fernanda", "Gilda", "Graziella",
        "Ines", "Irma", "Leda", "Letizia", "Liliana", "Margherita",
        "Milena", "Mirella", "Nadia", "Nicoletta", "Olga", "Palmira",
        "Pierina", "Renata", "Rita", "Romina", "Rosanna", "Sandra",
        "Tiziana", "Valeria", "Vanda", "Wanda", "Zelma",
        "Adalgisa", "Bruna", "Costanza", "Elvira", "Erminia",
        "Fortunata", "Geraldina", "Imelda", "Luigina", "Mafalda",
    ],
    "gr": [
        "Maria", "Eleni", "Sofia", "Katerina", "Despina", "Georgia",
        "Anastasia", "Dimitra", "Vassiliki", "Christina", "Ioanna",
        "Theodora", "Paraskevi", "Angeliki", "Fotini", "Panagiota",
        "Stavroula", "Kalliopi", "Evangelia", "Chrysoula", "Eirini",
        "Magdalini", "Alexandra", "Athina", "Konstantina",
        "Androniki", "Argyro", "Charikleia", "Danai", "Eftychia",
        "Fevronia", "Galatea", "Ioakima", "Kassiani", "Lamprini",
        "Melpomeni", "Niki", "Olympia", "Polyxeni", "Smaragda",
        "Styliani", "Triantafyllia", "Vasiliki", "Xanthippi", "Zoi",
        "Agapi", "Dimitra", "Hariklia", "Ifigenia", "Kalomira",
        # Additional
        "Aglaia", "Aspasia", "Areti", "Chaido", "Chrysanthi",
        "Dafni", "Diamanto", "Efrosini", "Erasmia", "Evdokia",
        "Filothei", "Garifalia", "Glykeria", "Ioanna", "Isidora",
        "Kalliopi", "Kyriaki", "Loukia", "Marika", "Marianthi",
        "Natassa", "Ourania", "Pagona", "Persefoni", "Pinelopi",
        "Roxani", "Soultana", "Stavrina", "Thaleia", "Toula",
        "Vagia", "Venetia", "Voula", "Yiota", "Zacharoula",
        "Agathi", "Antigoni", "Artemis", "Chrysovalantou", "Eleftheria",
        "Foteini", "Grammatiki", "Iosifina", "Kornilia", "Mersini",
    ],
    "vn": [
        "Mai", "Lan", "Huong", "Thuy", "Anh", "Linh", "Hoa", "Ngoc",
        "Trang", "Hanh", "Phuong", "Dung", "Thu", "Yen", "Chi", "Ngan",
        "Quynh", "Van", "Trinh", "Ly", "Dao", "Kim", "My", "Uyen", "Hang",
        "Bich", "Cam", "Diem", "Giang", "Ha", "Hien", "Hong", "Khanh",
        "Lien", "Minh", "Nhu", "Oanh", "Que", "Suong", "Tam",
        "Thanh", "Thao", "Thi", "Tuyet", "Vi", "Xuan", "Vy",
        "Chau", "Duyen", "Nhung",
        # Additional
        "An", "Bao", "Cuc", "Dan", "Diep", "Ha", "Hau", "Hue",
        "Huyen", "Kieu", "Le", "Loan", "Luu", "Man", "Muoi",
        "Nga", "Nhi", "Nhu", "Nu", "Pha", "Phung", "Quyen",
        "Sen", "Son", "Tam", "Thien", "Thom", "Thu", "Tien",
        "Tin", "Tram", "Trieu", "Truc", "Tu", "Tuong", "Ut",
        "Vui", "Xuan", "Yen", "Ai", "Chuyen", "Doan", "Hanh",
        "Hieu", "Lam", "Loc", "Ngat", "Phi", "Tho", "Tru", "Xuyen",
    ],
    "other_eu": [
        "Ana", "Inês", "Astrid", "Sofia", "Zsófia", "Maja", "Isabel",
        "Beatriz", "Carmen", "Elena", "Alina", "Ioana", "Marta", "Klára",
        "Tereza", "Hana", "Katarína", "Petra", "Elina", "Sigrid", "Ingrid",
        "Linnéa", "Ilona", "Márta", "Zsuzsanna", "Adriana", "Cristina",
        "Daniela", "Ruxandra", "Ágnes",
        "Birgitta", "Dagny", "Ebba", "Frida", "Gunnhild", "Hedvig",
        "Ildikó", "Judit", "Katalin", "Lujza", "Magdolna", "Noémi",
        "Orsolya", "Piroska", "Réka", "Sára", "Tünde", "Valéria",
        "Luminița", "Mihaela", "Nicoleta", "Oana", "Simona", "Tatiana",
        "Květoslava", "Milada", "Růžena", "Věra", "Blanka", "Drahomíra",
        # Additional
        "Anabela", "Branca", "Catarina", "Dulce", "Elvira", "Fernanda",
        "Graça", "Helena", "Idalina", "Lídia", "Manuela", "Odete",
        "Palmira", "Raquel", "Solange", "Teresa", "Britt", "Elisabet",
        "Gunilla", "Kerstin", "Margareta", "Marianne", "Sonja", "Ulla",
        "Beáta", "Enikő", "Erzsébet", "Hajnalka", "Nóra", "Orsolya",
        "Anamaria", "Cornelia", "Doina", "Elena", "Florentina", "Gabriela",
        "Ionela", "Lavinia", "Mirela", "Petronela", "Rodica", "Violeta",
        "Dagmar", "Ivana", "Jiřina", "Libuše", "Miroslava", "Vlasta",
    ],
}

SURNAMES: dict[str, list[str]] = {
    "de": [
        # Common
        "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
        "Becker", "Hoffmann", "Schäfer", "Koch", "Richter", "Klein", "Wolf",
        "Schröder", "Neumann", "Zimmermann", "Krüger", "Hartmann", "Lange",
        "Werner", "Braun", "Schmitt", "Krause", "Lehmann", "Schulz", "Maier",
        "Köhler", "Herrmann", "König", "Walter", "Huber", "Kaiser", "Fuchs",
        "Peters", "Scholz", "Lang", "Jung", "Möller", "Hahn", "Weiß",
        "Engel", "Vogel", "Keller", "Brandt", "Roth", "Frank", "Winter",
        "Sommer", "Ernst", "Baumann", "Haas", "Schreiber", "Kraft", "Berger",
        "Seidel", "Vogt", "Böhm", "Stein", "Horn", "Graf", "Busch", "Bergmann",
        # Less common / ambiguous / regional
        "Albrecht", "Arnold", "Bader", "Barthel", "Beyer", "Böttcher",
        "Brinkmann", "Büttner", "Decker", "Dietrich", "Dörr", "Ebert",
        "Eckert", "Fiedler", "Förster", "Franke", "Freitag", "Frey",
        "Geiger", "Gerber", "Greiner", "Groß", "Günther", "Haase",
        "Hammer", "Hanke", "Hein", "Heß", "Hilbert", "Jäger",
        "Janssen", "Kästner", "Keil", "Kirchner", "Kluge", "Knoll",
        "Kopp", "Kunz", "Kurz", "Langer", "Lenz", "Lindner",
        "Mai", "Marx", "Meister", "Nagel", "Naumann", "Noll",
        "Pfeiffer", "Raab", "Reich", "Reuter", "Römer", "Rudolph",
        "Schiller", "Schlegel", "Schuster", "Schwab", "Schwarz",
        "Seifert", "Singer", "Stahl", "Stark", "Straub", "Thiel",
        "Ulrich", "Unger", "Voigt", "Wahl", "Wegner", "Winkler",
        "Witt", "Zeller", "Ziegler",
        # Additional rare / regional / archaic
        "Abendroth", "Ackermann", "Adler", "Altmann", "Bachmann", "Bär",
        "Baumgartner", "Behrens", "Bertram", "Birkner", "Blank", "Blume",
        "Bode", "Born", "Brandl", "Breuer", "Brückner", "Buchholz",
        "Burghardt", "Claus", "Conrad", "Dahl", "Dittrich", "Döring",
        "Dreher", "Dressler", "Eichhorn", "Elsner", "Erdmann", "Falk",
        "Faust", "Fehling", "Felber", "Fleischer", "Foth", "Freund",
        "Fritsch", "Fuchs", "Funk", "Geißler", "Gerlach", "Glaser",
        "Gorski", "Grafe", "Grimm", "Groth", "Grundmann", "Hagedorn",
        "Harms", "Haupt", "Hecht", "Heil", "Heinze", "Held", "Henkel",
        "Hennig", "Herbst", "Herzog", "Heuer", "Hildebrandt", "Hirsch",
        "Höfer", "Holz", "Hummel", "Ihle", "Jahn", "Jakobs", "Jost",
        "Kahl", "Kamm", "Karg", "Kästner", "Kern", "Kiefer", "Klemm",
        "Klinger", "Klose", "Knauer", "Knecht", "Knorr", "Kolb",
        "Konrad", "Korn", "Kramer", "Krebs", "Krug", "Kuhn", "Kunze",
        "Kurth", "Lampe", "Landgraf", "Laube", "Lauer", "Lechner",
        "Liedtke", "Link", "Lohmann", "Lorenz", "Ludwig", "Lüdtke",
        "Mann", "Marquardt", "Matzke", "Mauer", "Merkel", "Merz",
        "Metz", "Michel", "Mohn", "Morgenstern", "Moser", "Münch",
        "Nickel", "Nolte", "Oertel", "Opitz", "Ott", "Pahl",
        "Palm", "Pape", "Petzold", "Pieper", "Platz", "Pohl", "Probst",
        "Rauch", "Rausch", "Rehm", "Reinhardt", "Renner", "Riedel",
        "Ritter", "Rose", "Rüger", "Ruppert", "Sachse", "Sauer",
        "Schade", "Schenk", "Schick", "Schindler", "Schlenker",
        "Schlüter", "Schmid", "Schnabel", "Scholze", "Schön", "Schulte",
        "Schulze", "Schwarz", "Seibert", "Siebert", "Sonntag", "Specht",
        "Stadler", "Stamm", "Stange", "Steiner", "Stelzer", "Stock",
        "Stolz", "Storm", "Strauß", "Strobel", "Sturm", "Tauber",
        "Thiele", "Tiedemann", "Trautmann", "Tremmel", "Vetter", "Voss",
        "Wacker", "Walther", "Weidner", "Wendt", "Wenzel", "Wichmann",
        "Wiedemann", "Wild", "Wimmer", "Wittig", "Wolff", "Wunderlich",
        "Zander", "Zehnder", "Zeller", "Zenker", "Zink", "Zorn",
    ],
    "tr": [
        "Yılmaz", "Demir", "Şahin", "Çelik", "Yıldız", "Öztürk", "Aydın", "Kaya",
        "Polat", "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Korkmaz",
        "Aktaş", "Karaca", "Yalçın", "Özdemir", "Güneş", "Koç", "Erdoğan",
        "Bulut", "Tekin", "Kaplan", "Turan", "Başar", "Acar", "Uçar", "Sezer",
        "Güler", "Keskin", "Taş", "Yavuz", "Ceylan", "Duran", "Özkan",
        "Karataş", "Yüksel", "Kara", "Kurt", "Zengin", "Balcı", "Coşkun",
        "Ateş", "Bayrak", "Duman", "Genç", "Gündüz", "Işık",
        "Kurucu", "Sönmez", "Tanrıverdi", "Toprak", "Uysal",
        "Yıldırım", "Bozkurt", "Demirtaş", "Gürbüz", "Sarı",
        # Additional
        "Akbaş", "Akkaya", "Altın", "Altıntaş", "Arıkan", "Avşar",
        "Aygün", "Bal", "Başaran", "Bayar", "Bilgin", "Bingöl",
        "Bircan", "Bozdemir", "Büyük", "Candemir", "Çakır", "Çam",
        "Çınar", "Dağ", "Dalkılıç", "Dikmen", "Dinç", "Dizdar",
        "Elmas", "Erdem", "Ergin", "Ertürk", "Eser", "Gök",
        "Gökçe", "Günaydın", "Güngör", "Gürkan", "Hatipoğlu", "İnan",
        "Kalender", "Kan", "Karaman", "Kılınç", "Kocabaş", "Koçak",
        "Köse", "Küçük", "Mutlu", "Öz", "Özcan", "Özer", "Özgür",
        "Parlak", "Peker", "Savaş", "Serdar", "Sevim", "Solak",
        "Şen", "Şimşek", "Tan", "Taşkın", "Teker", "Tok", "Tunç",
        "Türk", "Ulusoy", "Ünal", "Üstün", "Uzun", "Yaşar", "Yazıcı",
    ],
    "ar": [
        "Al-Sayed", "Haddad", "Khalil", "Nasser", "Aziz", "Mansour", "Saleh",
        "Ibrahim", "Hassan", "Ali", "Ahmad", "Mahmoud", "Yousef", "Abdallah",
        "Farouk", "Khoury", "Habib", "Darwish", "Bakir", "Kassem", "Hamdan",
        "Moussa", "Shaker", "Sabbagh", "Taha", "Saad", "Issa", "Jaber",
        "Awad", "Barakat",
        "Al-Masri", "Al-Rashid", "Benali", "Bouzid", "Chaabi", "Daoudi",
        "El-Amin", "Ghali", "Hariri", "Idrissi", "Jaafar", "Karimi",
        "Lachiri", "Mahdi", "Najjar", "Omari", "Qaderi", "Rahmani",
        "Samadi", "Tahiri", "Wahbi", "Zeroual", "Amrani", "Bekkali",
        "Cherkaoui", "Fassi", "Guedira", "Hajji", "Kettani", "Lamrani",
        # Additional
        "Abdel-Rahman", "Abou-Khalil", "Al-Amiri", "Al-Assadi", "Al-Azawi",
        "Al-Badri", "Al-Dabbagh", "Al-Fayed", "Al-Hakim", "Al-Husseini",
        "Al-Jaberi", "Al-Khateeb", "Al-Maliki", "Al-Nouri", "Al-Rawi",
        "Al-Saadi", "Al-Shamsi", "Al-Zubaidi", "Assaf", "Attar",
        "Bishara", "Boualem", "Dagher", "Dib", "Fares", "Ghanem",
        "Hadid", "Hammoudi", "Jibril", "Kanaan", "Mallah", "Matar",
        "Mikhail", "Naji", "Nasrallah", "Obeid", "Qassem", "Rizk",
        "Saba", "Samaha", "Sayegh", "Shamoun", "Turk", "Yacoub",
        "Zahra", "Zein", "Atiyah", "Bishay", "Diab", "Halabi",
    ],
    "pl": [
        "Nowak", "Kowalski", "Wiśniewski", "Wójcik", "Kamiński", "Lewandowski",
        "Zieliński", "Szymański", "Woźniak", "Dąbrowski", "Kozłowski",
        "Jankowski", "Mazur", "Kwiatkowski", "Krawczyk", "Pawlak", "Michalak",
        "Adamczyk", "Dudek", "Pietrzak", "Grabowski", "Sikora", "Walczak",
        "Baran", "Zawadzki", "Rutkowski", "Jaworski", "Stępień", "Olejnik",
        "Jabłoński",
        "Baranowski", "Chmielewski", "Czarnecki", "Dobrowolski", "Gajewski",
        "Jakubowski", "Kaczmarek", "Laskowski", "Marciniak", "Nowicki",
        "Olszewski", "Piątek", "Rogowski", "Sobczak", "Tomaszewski",
        "Urbaniak", "Witkowski", "Zakrzewski", "Borkowski", "Górniak",
        "Konieczny", "Lipiński", "Majewski", "Niedzielski", "Sokołowski",
        "Szczepański", "Wieczorek", "Bednarski", "Cieślak", "Głowacki",
        # Additional
        "Białek", "Bochenek", "Buczek", "Bujak", "Chojnacki", "Czerwiński",
        "Domagała", "Filipek", "Flis", "Gałka", "Gaweł", "Grzelak",
        "Grzesiak", "Jędrzejczyk", "Kaczor", "Kalinowski", "Klimek",
        "Kopeć", "Kowal", "Kozioł", "Kucharczyk", "Kurek", "Lis",
        "Łuczak", "Malec", "Matuszewski", "Mielczarek", "Mróz",
        "Mucha", "Muszyński", "Nowiński", "Owczarek", "Paluch",
        "Pawlik", "Pluta", "Polak", "Przybylski", "Rakowski",
        "Sawicki", "Serafin", "Skiba", "Sosnowski", "Sroka",
        "Stanek", "Sułkowski", "Szewczyk", "Tkaczyk", "Tomczak",
        "Tracz", "Turowski", "Wawrzyniak", "Więckowski", "Wilk",
        "Witek", "Włodarczyk", "Wojtas", "Wróblewski", "Wysocki",
        "Zając", "Zalewski", "Ziółkowski", "Żak", "Żurek",
    ],
    "ru": [
        "Iwanow", "Petrow", "Smirnow", "Wolkow", "Kuznetsow", "Popow", "Sokolow",
        "Fjodorow", "Morosow", "Jegorow", "Orlow", "Wassiljew", "Lebedew",
        "Koslow", "Nowikow", "Sacharow", "Pawlow", "Stepanow", "Golubew",
        "Winogradow", "Bogdanow", "Woronow", "Medwedew", "Grigorjew",
        "Romanow", "Tarassow", "Makarow", "Nikitin", "Beljakow", "Tschernow",
        "Bauer", "Braun", "Keller", "Kraft", "Schäfer", "Schneider",
        "Fischer", "Schwarz", "Weiss", "Herz", "Ritter", "Steinbach",
        "Kling", "Voss", "Diehl", "Bachmann", "Fuchs", "Fink",
        "Ossipow", "Schabalow", "Denikin", "Krawtschenko", "Kondratjew",
        "Prochorow", "Samsonow", "Tkatschenko", "Kowalenko", "Bondarenko",
        "Gontscharow", "Schewchenko",
        # Additional
        "Abramow", "Aksjonow", "Beljajew", "Bulanow", "Danilow",
        "Dawydow", "Frolow", "Galkow", "Gerassimow", "Gladkow",
        "Gorbunow", "Gromow", "Jakowlew", "Karpow", "Kirillow",
        "Kolesnikow", "Komarow", "Konowalow", "Kotow", "Krylow",
        "Larionow", "Lasarew", "Loginow", "Lykow", "Malkow",
        "Martynow", "Maximow", "Melnikow", "Mironow", "Nasarow",
        "Nesterow", "Nikulin", "Osipow", "Panow", "Pestow",
        "Platonow", "Poljakow", "Rjabinin", "Rogatschow", "Scharow",
        "Schukow", "Seliwanow", "Sidorow", "Subbotin", "Suslow",
        "Titow", "Tscherepanow", "Uschakow", "Wawilow", "Wlassow",
        "Worobjow", "Wysozkij", "Zaitsew", "Zukow", "Zyganow",
    ],
    "ex_yu": [
        "Jovanović", "Petrović", "Horvat", "Babić", "Marković", "Knežević",
        "Nikolić", "Pavlović", "Todorović", "Stojanović", "Kovačević",
        "Ilić", "Popović", "Stanković", "Milošević", "Savić", "Ristić",
        "Krstić", "Obradović", "Živković", "Vuković", "Radovanović",
        "Bogdanović", "Matić", "Tomić", "Lukić", "Đorđević", "Simić",
        "Kostić", "Janković",
        "Arslanagić", "Begović", "Ćosić", "Delalić", "Filipović",
        "Golubović", "Hadžić", "Ivanović", "Jurić", "Krajišnik",
        "Lazarević", "Mijatović", "Novaković", "Orlović", "Pejić",
        "Rajković", "Šarić", "Terzić", "Vasiljević", "Zlatić",
        "Agić", "Bjelić", "Čolić", "Dragić", "Erić",
        "Grbić", "Hrnjak", "Ibrišimović", "Jeremić", "Karadžić",
        # Additional
        "Andrić", "Balić", "Blažević", "Bošnjak", "Brkić", "Bulatović",
        "Cvijović", "Čvorović", "Delić", "Dimić", "Došen", "Đukić",
        "Galić", "Gavrilović", "Grujić", "Halilović", "Husejnović",
        "Isaković", "Jakšić", "Jelić", "Kalajdžić", "Katić",
        "Kojić", "Kuzmanović", "Lalić", "Mandić", "Marić",
        "Meštrović", "Milić", "Milovanović", "Mirković", "Mujić",
        "Mušić", "Nedić", "Ninković", "Pantelić", "Perić",
        "Petković", "Radić", "Salkić", "Savović", "Spasojevic",
        "Stevanović", "Šljivić", "Tadić", "Vlahović", "Vučić",
        "Vujić", "Vukašinović", "Zečević", "Živanović", "Zorić",
    ],
    "it": [
        "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Greco",
        "Colombo", "Ricci", "Marino", "Bruno", "Gallo", "Conti", "De Luca",
        "Mancini", "Costa", "Giordano", "Rizzo", "Lombardi", "Moretti",
        "Barbieri", "Fontana", "Santoro", "Marini", "Rinaldi", "Caruso",
        "Ferrara", "Galli", "Martini", "Leone",
        "Amato", "Barone", "Cattaneo", "D'Angelo", "Ferraro", "Gentile",
        "Longo", "Marchetti", "Neri", "Orlando", "Palmieri", "Quattrocchi",
        "Ruggiero", "Sorrentino", "Testa", "Valentini", "Vitale", "Zanetti",
        "Benedetti", "Caputo", "De Angelis", "Fabbri", "Guerra", "Messina",
        "Pellegrini", "Rizzi", "Serra", "Villa", "Basile", "Coppola",
        # Additional
        "Aiello", "Arena", "Battaglia", "Bellini", "Bernardi", "Calabrese",
        "Campagna", "Capelli", "Catalano", "Ceccarelli", "Conte", "Corti",
        "Crespi", "D'Amico", "De Rosa", "Del Vecchio", "Di Stefano",
        "Donati", "Ferro", "Fiore", "Forte", "Fumagalli", "Garofalo",
        "Gatti", "Grasso", "Guerrini", "Iannone", "La Rosa", "Landi",
        "Leoni", "Lo Presti", "Maffei", "Mazzola", "Monti", "Morelli",
        "Negri", "Olivieri", "Pace", "Pagano", "Parisi", "Piras",
        "Poli", "Pugliese", "Raimondi", "Rossetti", "Sala", "Sanna",
        "Silvestri", "Spinelli", "Taverna", "Venturi", "Volpe", "Zanini",
    ],
    "gr": [
        "Papadopoulos", "Nikolaidis", "Georgiou", "Vasileiou", "Pappas",
        "Alexiou", "Christodoulou", "Dimitriou", "Economou", "Ioannidis",
        "Karagiannis", "Konstantinidis", "Makris", "Papageorgiou",
        "Theodoridis", "Vlachos", "Panagiotou", "Stavridis", "Zachariadis",
        "Diamantidis", "Antoniou", "Athanasiadis", "Bakoyannis",
        "Galanopoulos", "Karamanolis",
        "Anastasiou", "Christoforou", "Deligiannis", "Efthymiou",
        "Floros", "Giannakopoulos", "Hatzidakis", "Kalogeropoulos",
        "Kyriazis", "Liakopoulos", "Mavridis", "Nikolopoulos",
        "Papadimitriou", "Raptis", "Sideris", "Tsalikis",
        "Vassilopoulos", "Xenos", "Zafeiris", "Argyropoulos",
        "Bouloukos", "Chatzigeorgiou", "Dritsas", "Eleftheriou",
        "Georgakopoulos",
        # Additional
        "Adamopoulos", "Anagnostopoulos", "Andreopoulos", "Angelopoulos",
        "Apostolakis", "Avramidis", "Balaskas", "Charitidis",
        "Chronopoulos", "Daskalakis", "Fotopoulos", "Hadjipavlou",
        "Kaloudis", "Kapetanakis", "Katsaros", "Kondylis",
        "Koutsogiannis", "Lamprou", "Lekkas", "Mantzaris",
        "Markopoulos", "Matthaiou", "Mitropoulos", "Moschonas",
        "Nikopoulos", "Oikonomidis", "Panagopoulos", "Papathanasiou",
        "Samaras", "Siafakas", "Skouras", "Sotiropoulos",
        "Spiliopoulos", "Stamatiou", "Symeonidis", "Tasoulas",
        "Trikilis", "Tsoukalas", "Vasilakopoulos", "Verginis",
        "Voulgaris", "Xanthopoulos", "Zafiris", "Zervos", "Zogas",
    ],
    "vn": [
        "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi",
        "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Trịnh", "Đinh", "Lương",
        "Tô", "Phan", "Mai", "Vương", "Tạ", "Cao", "Châu", "Lâm", "Hà",
        "Tăng", "Từ", "Thái", "Quách", "Huỳnh", "Võ", "Diệp", "Trương",
        "Hứa", "La", "Mạc", "Nghiêm", "Ông", "Phùng", "Tống",
        "Ung", "Viên", "Kiều", "Đoàn", "Lục", "Triệu", "Khương",
        "Sơn", "Cung", "Giáp",
        # Additional
        "An", "Bạch", "Biện", "Cầm", "Chung", "Cù", "Đào", "Đậu",
        "Doãn", "Dư", "Giang", "Hạ", "Hàn", "Khổng", "Khuất",
        "Kiểu", "Kim", "Lạc", "Lại", "Lâu", "Liễu", "Lục",
        "Lưu", "Mã", "Mạch", "Mao", "Ninh", "Nông", "Ôn",
        "Phí", "Quản", "Sái", "Tào", "Thạch", "Thân", "Thiều",
        "Tiêu", "Tôn", "Trà", "Văn", "Vi", "Vĩnh", "Vụ",
        "Xuân", "Yên",
    ],
    "other_eu": [
        "Silva", "Santos", "Andersson", "Nilsson", "Nagy", "Kovács",
        "Ferreira", "Oliveira", "Johansson", "Larsson", "Szabó", "Tóth",
        "Pereira", "Almeida", "Eriksson", "Persson", "Horváth", "Varga",
        "Rodríguez", "García", "Martínez", "Fernández", "Popescu", "Ionescu",
        "Novák", "Dvořák", "Černý", "Kučera", "Havel", "Bartoš",
        "Lindqvist", "Bergström", "Nyström", "Sjöberg", "Hedlund",
        "Pálinkás", "Balogh", "Fekete", "Fehér", "Molnár",
        "Soares", "Carvalho", "Ribeiro", "Monteiro", "Gonçalves",
        "Dumitrescu", "Gheorghe", "Marinescu", "Stoica", "Popa",
        "Horák", "Svoboda", "Procházka", "Veselý", "Kolář",
        "Stanković", "Obrenović", "Petrescu", "Bogdanov", "Dimov",
        # Additional
        "Abrahamsson", "Blomqvist", "Dahlberg", "Ekström", "Forsberg",
        "Gustafsson", "Holmberg", "Isaksson", "Jakobsson", "Karlsson",
        "Lindberg", "Lundberg", "Magnusson", "Nordin", "Olsson",
        "Pettersson", "Sandberg", "Ström", "Sundberg", "Wallin",
        "Bognár", "Csizmadia", "Farkas", "Gulyás", "Hegedüs",
        "Juhász", "Kelemen", "Kiss", "Lakatos", "Papp",
        "Barbosa", "Costa", "Dias", "Fernandes", "Lima", "Lopes",
        "Machado", "Martins", "Mendes", "Moreira", "Rodrigues", "Sousa",
        "Ciobanu", "Constantinescu", "Dinu", "Enescu", "Florescu",
        "Lazarescu", "Munteanu", "Neagu", "Olaru", "Radu",
        "Beneš", "Fiala", "Holub", "Jelínek", "Král", "Marek",
        "Němec", "Pokorný", "Richter", "Růžička", "Sedláček",
        "Šimek", "Šťastný", "Urban", "Vaněk", "Vlček",
    ],
}
# fmt: on

# Nobility particles (rare, but a known hard case for name detection).
NOBILITY_PARTICLES = ["von", "von der", "zu", "von und zu"]

# --- Adversarial name pools ------------------------------------------------
# These pools are used exclusively by the adversarial difficulty tier.

# Surnames that are also medical eponyms — forces disambiguation between
# "Morbus Crohn" (don't redact) and "Frau Crohn" (redact).
# fmt: off
EPONYM_SURNAMES: list[str] = [
    "Crohn", "Parkinson", "Cushing", "Hodgkin", "Addison", "Hashimoto",
    "Raynaud", "Marfan", "Basedow", "Bell", "Sjögren", "Wegener",
    "Kaposi", "Bowen", "Burkitt", "Paget", "Behçet", "Reiter",
    "Korsakow", "Turner", "Kawasaki", "Babinski", "Murphy", "Romberg",
    "Trendelenburg", "Lachman", "Killip", "Braden", "Norton", "Apgar",
    "Virchow", "Langerhans", "Charcot", "Douglas",
]

# Surnames that are common German words — appear naturally in clinical text.
# "Fischer" the patient vs "fischt" in compound words; "Koch" vs "kochen", etc.
WORD_SURNAMES: list[str] = [
    "Koch", "Fischer", "Schwarz", "Lang", "Klein", "Groß", "Richter",
    "Sommer", "Winter", "Engel", "Stark", "Kurz", "Schön", "Braun",
    "Wolf", "Fuchs", "Hahn", "Adler", "Vogel", "Roth", "Ernst",
    "Kraus", "Busch", "Stein", "Graf", "Fink", "Kohl", "Reis",
    "Stern", "Weiß", "Schwab", "Franke", "Bach", "Berg", "Hirt",
    "Kraft", "Licht", "Nagel", "Rauch", "Strauch", "Hase", "Barth",
]

# Mapping of word-surnames to natural German text fragments that use the word
# in various forms (to create "Fischers Fritz fischt frische Fische" situations).
WORD_SURNAME_CONTEXTS: dict[str, list[str]] = {
    "Koch": ["kochen", "kochte", "gekocht", "Kochsalzlösung", "Kochfeld"],
    "Fischer": ["fischt", "Fischerei", "fischig", "Fischöl"],
    "Schwarz": ["schwarz", "schwarze", "schwarzem", "Schwarzwald"],
    "Lang": ["lang", "lange", "langen", "langfristig", "langjährig", "langwierig"],
    "Klein": ["klein", "kleine", "kleinen", "kleinem", "kleinzellig", "Kleinhirn"],
    "Groß": ["groß", "große", "großen", "großem", "großzelliges", "Großhirn"],
    "Richter": ["richtet", "gerichtet", "aufgerichtet", "eingerichtet"],
    "Sommer": ["sommerlich", "Sommerzeit", "Sommergrippe"],
    "Winter": ["winterlich", "Wintermonate", "überwintert"],
    "Engel": ["engelartig"],
    "Stark": ["stark", "starke", "starken", "starkem", "verstärkt", "Verstärkung"],
    "Kurz": ["kurz", "kurze", "kurzen", "kurzfristig", "Kurzatmigkeit", "kürzlich"],
    "Schön": ["schön", "schöne", "schönem", "schonend", "Schonung"],
    "Wolf": ["Wolfsrachen"],
    "Fuchs": ["Fuchsbandwurm"],
    "Hahn": ["Hahnentritt"],
    "Vogel": ["Vogelgrippe"],
    "Stern": ["Sternalpunktion", "Sternum"],
    "Weiß": ["weiß", "weiße", "weißen", "weißem", "weißlich", "Weißkittel"],
    "Kraft": ["kraft", "Kraft", "kräftig", "kraftlos", "Muskelkraft", "Handkraft"],
    "Stein": ["Stein", "steinig", "Nierenstein", "Gallenstein", "Steinschnitt"],
    "Berg": ["bergab", "Herzberg", "Eisberg"],
    "Nagel": ["Nagel", "Nagelbett", "Nagelfalz", "Nagelpsoriasis"],
    "Rauch": ["Rauch", "Raucher", "Raucherstatus", "Nichtraucher"],
    "Bach": ["Bachblüten"],
}

# Surreal / hard-to-recognize names — very short, look like abbreviations,
# or have unusual character patterns for German clinical context.
SURREAL_GIVEN_NAMES_M: list[str] = [
    "Li", "Wu", "Bo", "Yi", "Ai", "Ng", "Ho", "Jo", "Lo", "Xu",
    "Pi", "Qi", "Vo", "An", "Ed", "Ib", "Oz", "Uy", "If", "Op",
]
SURREAL_GIVEN_NAMES_F: list[str] = [
    "Li", "Yi", "Ai", "Su", "Mi", "Na", "Qi", "Xi", "Ju", "Lu",
    "Bi", "Di", "Fi", "Gi", "Ni", "Pi", "Ti", "Zi", "Vu", "Io",
]
SURREAL_SURNAMES: list[str] = [
    "Ng", "Wu", "Li", "Ho", "Ma", "Hu", "Xu", "Qi", "Yi", "Bo",
    "Pho", "Thi", "Bui", "Lam", "Ngo", "Cao", "Vu", "Do", "Le", "Ly",
    "Oz", "Ur", "Ak", "Er", "Il", "Ok", "Un", "As", "Is", "Os",
]
# fmt: on


def sample_eponym_surname(rng: random.Random) -> str:
    """Pick a surname that is also a medical eponym."""
    return rng.choice(EPONYM_SURNAMES)


def sample_word_surname(rng: random.Random) -> str:
    """Pick a surname that is a common German word."""
    return rng.choice(WORD_SURNAMES)


def sample_surreal_name(rng: random.Random, gender: str | None = None) -> SampledName:
    """Sample a surreal/hard-to-recognize name (very short, ambiguous)."""
    if gender is None:
        gender = rng.choice(["m", "f"])
    given = rng.choice(
        SURREAL_GIVEN_NAMES_M if gender == "m" else SURREAL_GIVEN_NAMES_F
    )
    nachname = rng.choice(SURREAL_SURNAMES)
    return SampledName(
        vorname=given, nachname=nachname, gender=gender, origin="other_eu"
    )


@dataclass(frozen=True)
class SampledName:
    vorname: str
    nachname: str
    gender: str  # "m" | "f"
    origin: str


def _weighted_origin(
    rng: random.Random, origin_weights: dict[str, float] | None = None
) -> str:
    ow = origin_weights if origin_weights is not None else ORIGIN_WEIGHTS
    origins = list(ow.keys())
    weights = list(ow.values())
    return rng.choices(origins, weights=weights, k=1)[0]


def sample_name(
    rng: random.Random,
    gender: str | None = None,
    *,
    allow_double: bool = True,
    allow_nobility: bool = True,
    difficulty: str = "medium",
    cfg: "Any | None" = None,
) -> SampledName:
    """Sample one demographically-grounded name.

    Surname origin may differ from given-name origin (mixed families), which
    adds realistic variation. Optionally produces double given names and
    nobility particles as hard cases.

    Difficulty controls double-name frequency:
      easy:   ~8% double given, ~5% double surname
      medium: ~12% double given, ~8% double surname
      hard:   ~22% double given, ~15% double surname, cross-origin surnames forced
              ~10% surreal/very-short names
    """
    # Config knobs (fall back to the historical literals when unconfigured).
    origin_weights = cfg.origin_weights if cfg is not None else ORIGIN_WEIGHTS
    surreal_p = cfg.surreal_prob_hard if cfg is not None else 0.10
    cross_map = (
        cfg.cross_origin_p
        if cfg is not None
        else {"easy": 0.00, "medium": 0.00, "hard": 0.40}
    )
    double_given_map = (
        cfg.double_given_p
        if cfg is not None
        else {"easy": 0.00, "medium": 0.12, "hard": 0.22}
    )
    double_surname_map = (
        cfg.double_surname_p
        if cfg is not None
        else {"easy": 0.00, "medium": 0.08, "hard": 0.15}
    )
    nobility_p = cfg.nobility_prob if cfg is not None else 0.03

    if gender is None:
        gender = rng.choice(["m", "f"])

    # Hard: ~10% chance of surreal/very-short name
    if difficulty == "hard" and rng.random() < surreal_p:
        return sample_surreal_name(rng, gender)

    given_pool = GIVEN_NAMES_M if gender == "m" else GIVEN_NAMES_F

    given_origin = _weighted_origin(rng, origin_weights)
    # Hard: much higher chance of cross-origin given/surname combo
    cross_origin_p = cross_map.get(difficulty, 0.40)
    surname_origin = (
        given_origin
        if rng.random() > cross_origin_p
        else _weighted_origin(rng, origin_weights)
    )

    # Difficulty-scaled double-name probabilities.
    # Easy: no double/hyphenated names — keeps surface forms unambiguous.
    double_given_p = double_given_map.get(difficulty, 0.22)
    double_surname_p = double_surname_map.get(difficulty, 0.15)

    vorname = rng.choice(given_pool[given_origin])
    if allow_double and rng.random() < double_given_p:
        sep = rng.choice(["-", " "])
        second = rng.choice(given_pool[given_origin])
        if second != vorname:
            vorname = f"{vorname}{sep}{second}"

    nachname = rng.choice(SURNAMES[surname_origin])
    if allow_double and rng.random() < double_surname_p:
        # Hard: force cross-origin second surname for extra challenge
        if difficulty in ("hard", "adversarial"):
            other_origins = [o for o in origin_weights if o != surname_origin]
            second_origin = rng.choice(other_origins)
        else:
            second_origin = _weighted_origin(rng, origin_weights)
        second = rng.choice(SURNAMES[second_origin])
        if second != nachname:
            nachname = f"{nachname}-{second}"
    # ~3% nobility particle (German origin only)
    elif allow_nobility and surname_origin == "de" and rng.random() < nobility_p:
        nachname = f"{rng.choice(NOBILITY_PARTICLES)} {nachname}"

    return SampledName(
        vorname=vorname, nachname=nachname, gender=gender, origin=given_origin
    )
