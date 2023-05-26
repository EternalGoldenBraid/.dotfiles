import matplotlib.pyplot as plt

data = """
Akaa    12830    4549    35.5    4397    34.3    8946    69.7
Hämeenkyrö    8150    3222    39.5    2621    32.2    5843    71.7
Ikaalinen    5732    2345    40.9    1733    30.2    4078    71.1
Juupajoki    1521    620    40.8    432    28.4    1052    69.2
Kangasala    23999    8872    37.0    9093    37.9    17965    74.9
Kihniö    1586    712    44.9    449    28.3    1161    73.2
Lempäälä    16688    6160    36.9    6486    38.9    12646    75.8
Mänttä-Vilppula    8313    2876    34.6    2686    32.3    5562    66.9
Nokia    25570    10681    41.8    8259    32.3    18940    74.1
Orivesi    7436    3137    42.2    2258    30.4    5395    72.6
Parkano    5294    2292    43.3    1429    27.0    3721    70.3
Pirkkala    14404    6144    42.7    5137    35.7    11281    78.3
Punkalaidun    2366    919    38.8    828    35.0    1747    73.8
Pälkäne    5180    2049    39.6    1725    33.3    3774    72.9
Ruovesi    3675    1516    41.3    1140    31.0    2656    72.3
Sastamala    19696    7698    39.1    6354    32.3    14052    71.3
Tampere    188757    78574    41.6    62195    32.9    140769    74.6
Urjala    3911    1312    33.5    1363    34.9    2675    68.4
Valkeakoski    16536    5707    34.5    6033    36.5    11740    71.0
Vesilahti    3212    1229    38.3    1232    38.4    2461    76 6
Virrat    5593    2202    39.4    1681    30.1    3883    69.4
Ylöjärvi    24211    8737    36.1    9256    38.2    17993    74.3
"""

rows = data.strip().split('\n')
municipalities = []
total_voter_percentage = []

for row in rows:
    columns = row.split()
    municipality = columns[0]
    percentage = float(columns[-1])
    municipalities.append(municipality)
    total_voter_percentage.append(percentage)

# fig, ax = plt.subplots(figsize=(15, 6))
# ax.bar(municipalities, total_voter_percentage)
# ax.set_xlabel('Municipality')
# ax.set_ylabel('Voter Turnout Percentage')
# ax.set_title('Voter Turnout Percentage by Municipality')

# plt.xticks(rotation=45, ha='right')

# plt.show()

import matplotlib.pyplot as plt

data = """
Suomen Keskusta 	26536 	8.9 	2
Perussuomalaiset 	51819 	17.3 	4
Kansallinen Kokoomus 	55244 	18.5 	4
Suomen Sosialidemokraattinen Puolue 	66109 	22.1 	5
Vihreä liitto 	37152 	12.4 	2
Vasemmistoliitto 	24341 	8.1 	1
Suomen ruotsalainen kansanpuolue 	327 	0.1 	0
Suomen Kristillisdemokraatit (KD) 	17180 	5.7 	1
Suomen Kommunistinen Puolue 	728 	0.2 	0
Kommunistinen Työväenpuolue - Rauhan ja Sosialismin puolesta 	92 	0.0 	0
Liberaalipuolue - Vapaus valita 	703 	0.2 	0
Piraattipuolue 	2018 	0.7 	0
Eläinoikeuspuolue 	604 	0.2 	0
Kansalaispuolue 	2964 	1.0 	0
Feministinen puolue 	597 	0.2 	0
Itsenäisyyspuolue 	621 	0.2 	0
Sininen tulevaisuus 	3342 	1.1 	0
Suomen Kansa Ensin 	228 	0.1 	0
Seitsemän tähden liike 	1371 	0.5 	0
Liike Nyt ry Pirkanmaa yhteislista 	7178 	2.4 	0
"""

rows = data.strip().split('\n')
party_names = []
num_seats = []

for row in rows:
    columns = row.split()
    party_name = ' '.join(columns[:-3])
    seats = int(columns[-1])
    party_names.append(party_name)
    num_seats.append(seats)

fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(party_names, num_seats)
ax.set_xlabel('Party')
ax.set_ylabel('Number of Seats')
ax.set_title('Number of Seats per Party')

plt.xticks(rotation=45, ha='right')

plt.show()
