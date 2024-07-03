#!/usr/bin/env python
# coding: utf-8

# In[1]:


text='''328,48,521585,17-10-2014,OH,250/500,1000,1406.91,0,466132,MALE,MD,craft-repair,sleeping,husband,53300,0,25-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,SC,Columbus,9935 4th Drive,5,1,YES,1,2,YES,71610,6510,13020,52080,Saab,92x,2004,Y
228,42,342868,27-06-2006,IN,250/500,2000,1197.22,5000000,468176,MALE,MD,machine-op-inspct,reading,other-relative,0,0,21-01-2015,Vehicle Theft,?,Minor Damage,Police,VA,Riverwood,6608 MLK Hwy,8,1,?,0,0,?,5070,780,780,3510,Mercedes,E400,2007,Y
134,29,687698,06-09-2000,OH,100/300,2000,1413.14,5000000,430632,FEMALE,PhD,sales,board-games,own-child,35100,0,22-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Columbus,7121 Francis Lane,7,3,NO,2,3,NO,34650,7700,3850,23100,Dodge,RAM,2007,N
256,41,227811,25-05-1990,IL,250/500,2000,1415.74,6000000,608117,FEMALE,PhD,armed-forces,board-games,unmarried,48900,-62400,10-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,OH,Arlington,6956 Maple Drive,5,1,?,1,2,NO,63400,6340,6340,50720,Chevrolet,Tahoe,2014,Y
228,44,367455,06-06-2014,IL,500/1000,1000,1583.91,6000000,610706,MALE,Associate,sales,board-games,unmarried,66000,-46000,17-02-2015,Vehicle Theft,?,Minor Damage,None,NY,Arlington,3041 3rd Ave,20,1,NO,0,1,NO,6500,1300,650,4550,Accura,RSX,2009,N
256,39,104594,12-10-2006,OH,250/500,1000,1351.1,0,478456,FEMALE,PhD,tech-support,bungie-jumping,unmarried,0,0,02-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Arlington,8973 Washington St,19,3,NO,0,2,NO,64100,6410,6410,51280,Saab,95,2003,Y
137,34,413978,04-06-2000,IN,250/500,1000,1333.35,0,441716,MALE,PhD,prof-specialty,board-games,husband,0,-77000,13-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NY,Springfield,5846 Weaver Drive,0,3,?,0,0,?,78650,21450,7150,50050,Nissan,Pathfinder,2012,N
165,37,429027,03-02-1990,IL,100/300,1000,1137.03,0,603195,MALE,Associate,tech-support,base-jumping,unmarried,0,0,27-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,VA,Columbus,3525 3rd Hwy,23,3,?,2,2,YES,51590,9380,9380,32830,Audi,A5,2015,N
27,33,485665,05-02-1997,IL,100/300,500,1442.99,0,601734,FEMALE,PhD,other-service,golf,own-child,0,0,30-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,WV,Arlington,4872 Rock Ridge,21,1,NO,1,1,YES,27700,2770,2770,22160,Toyota,Camry,2012,N
212,42,636550,25-07-2011,IL,100/300,500,1315.68,0,600983,MALE,PhD,priv-house-serv,camping,wife,0,-39300,05-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,NC,Hillsdale,3066 Francis Ave,14,1,NO,2,1,?,42300,4700,4700,32900,Saab,92x,1996,N
235,42,543610,26-05-2002,OH,100/300,500,1253.12,4000000,462283,FEMALE,Masters,exec-managerial,dancing,other-relative,38400,0,06-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,NY,Northbend,1558 1st Ridge,22,1,YES,2,2,?,87010,7910,15820,63280,Ford,F150,2002,N
447,61,214618,29-05-1999,OH,100/300,2000,1137.16,0,615561,FEMALE,High School,exec-managerial,skydiving,other-relative,0,-51000,15-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Springfield,5971 5th Hwy,21,3,YES,1,2,YES,114920,17680,17680,79560,Audi,A3,2006,N
60,23,842643,20-11-1997,OH,500/1000,500,1215.36,3000000,432220,MALE,MD,protective-serv,reading,wife,0,0,22-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Northbend,6655 5th Drive,9,1,YES,1,0,NO,56520,4710,9420,42390,Saab,95,2000,N
121,34,626808,26-10-2012,OH,100/300,1000,936.61,0,464652,FEMALE,MD,armed-forces,bungie-jumping,wife,52800,-32800,08-01-2015,Parked Car,?,Minor Damage,None,SC,Springfield,6582 Elm Lane,5,1,NO,1,1,NO,7280,1120,1120,5040,Toyota,Highlander,2010,N
180,38,644081,28-12-1998,OH,250/500,2000,1301.13,0,476685,FEMALE,College,machine-op-inspct,board-games,not-in-family,41300,-55500,15-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,SC,Springfield,6851 3rd Drive,12,1,NO,0,2,YES,46200,4200,8400,33600,Dodge,Neon,2003,Y
473,58,892874,19-10-1992,IN,100/300,2000,1131.4,0,458733,FEMALE,MD,transport-moving,movies,other-relative,55700,0,29-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,WV,Hillsdale,9573 Weaver Ave,12,4,YES,0,0,NO,63120,10520,10520,42080,Accura,MDX,1999,Y
70,26,558938,08-06-2005,OH,500/1000,1000,1199.44,5000000,619884,MALE,College,machine-op-inspct,hiking,own-child,63600,0,22-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,NY,Riverwood,5074 3rd St,0,3,?,1,2,YES,52110,5790,5790,40530,Nissan,Maxima,2012,N
140,31,275265,15-11-2004,IN,500/1000,500,708.64,6000000,470610,MALE,High School,machine-op-inspct,reading,unmarried,53500,0,06-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,WV,Northbend,4546 Tree St,9,1,NO,0,2,YES,77880,14160,7080,56640,Suburu,Legacy,2015,N
160,37,921202,28-12-2014,OH,500/1000,500,1374.22,0,472135,FEMALE,MD,craft-repair,yachting,other-relative,45500,-37800,19-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Northbrook,3842 Solo Ridge,19,1,YES,1,0,NO,72930,6630,13260,53040,Accura,TL,2015,N
196,39,143972,02-08-1992,IN,500/1000,2000,1475.73,0,477670,FEMALE,High School,handlers-cleaners,camping,own-child,57000,-27300,22-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,VA,Columbus,8101 3rd Ridge,8,3,?,2,0,NO,60400,6040,6040,48320,Nissan,Pathfinder,2014,N
460,62,183430,25-06-2002,IN,250/500,1000,1187.96,4000000,618845,MALE,JD,other-service,bungie-jumping,own-child,0,0,01-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Columbus,5380 Pine St,20,3,NO,1,0,?,47160,0,5240,41920,Suburu,Impreza,2011,N
217,41,431876,27-11-2005,IL,500/1000,2000,875.15,0,442479,FEMALE,Associate,machine-op-inspct,skydiving,own-child,46700,0,10-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Arlington,8957 Weaver Drive,15,3,?,1,2,?,37840,0,4730,33110,Accura,RSX,1996,N
370,55,285496,27-05-1994,IL,100/300,2000,972.18,0,443920,MALE,High School,prof-specialty,paintball,other-relative,72700,-68200,11-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Hillsdale,2526 Embaracadero Ave,20,3,NO,0,0,YES,71520,17880,5960,47680,Suburu,Forrestor,2000,Y
413,55,115399,08-02-1991,IN,100/300,2000,1268.79,0,453148,MALE,MD,priv-house-serv,chess,own-child,0,-31000,19-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Northbend,5667 4th Drive,15,1,?,2,2,?,98160,8180,16360,73620,Dodge,RAM,2011,Y
237,40,736882,02-02-1996,IN,100/300,1000,883.31,0,434733,MALE,College,craft-repair,kayaking,husband,0,-53500,24-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,VA,Riverwood,2502 Apache Hwy,6,1,NO,1,3,NO,77880,7080,14160,56640,Ford,Escape,2005,N
8,35,699044,05-12-2013,OH,100/300,2000,1266.92,0,613982,MALE,Masters,sales,polo,own-child,0,0,09-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,OH,Arlington,3418 Texas Lane,16,3,NO,1,3,YES,71500,16500,11000,44000,Ford,Escape,2006,Y
257,43,863236,20-09-1990,IN,100/300,2000,1322.1,0,436984,MALE,High School,prof-specialty,golf,own-child,0,-29200,28-01-2015,Parked Car,?,Minor Damage,Police,PA,Arlington,2533 Elm St,4,1,YES,1,3,YES,9020,1640,820,6560,Toyota,Camry,2005,N
202,34,608513,18-07-2002,IN,100/300,500,848.07,3000000,607730,MALE,JD,exec-managerial,chess,not-in-family,31000,-30200,07-01-2015,Vehicle Theft,?,Minor Damage,None,VA,Northbrook,3790 Andromedia Hwy,5,1,YES,2,1,?,5720,1040,520,4160,Suburu,Forrestor,2003,Y
224,40,914088,08-02-1990,OH,100/300,2000,1291.7,0,609837,FEMALE,JD,sales,kayaking,not-in-family,0,-55600,08-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,SC,Northbend,3220 Rock Drive,21,1,NO,1,0,YES,69840,7760,15520,46560,Dodge,Neon,2009,N
241,45,596785,04-03-2014,IL,500/1000,2000,1104.5,0,432211,FEMALE,PhD,machine-op-inspct,basketball,unmarried,0,0,15-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Northbrook,2100 Francis Drive,5,1,NO,2,2,NO,91650,14100,14100,63450,Accura,TL,2011,N
64,25,908616,18-02-2000,IL,250/500,1000,954.16,0,473328,MALE,Masters,prof-specialty,video-games,husband,53200,0,18-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,SC,Columbus,4687 5th Drive,22,4,NO,0,0,?,75600,12600,12600,50400,Toyota,Corolla,2005,N
166,37,666333,19-06-2008,IL,100/300,2000,1337.28,8000000,610393,MALE,JD,craft-repair,reading,husband,27500,0,28-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,WV,Riverwood,9038 2nd Lane,10,3,NO,2,2,?,67140,7460,7460,52220,Ford,F150,2006,Y
155,35,336614,01-08-2003,IL,500/1000,1000,1088.34,0,614780,FEMALE,Associate,adm-clerical,yachting,other-relative,81100,0,24-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,NY,Arlington,6092 5th Ave,16,3,YES,2,3,NO,29790,3310,3310,23170,BMW,3 Series,2008,N
114,30,584859,04-04-1992,IL,100/300,1000,1558.29,0,472248,MALE,High School,farming-fishing,video-games,wife,51400,-64000,09-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Hillsdale,8353 Britain Ridge,1,3,NO,1,2,?,77110,14020,14020,49070,Suburu,Impreza,2015,N
149,37,990493,13-01-1991,IL,500/1000,500,1415.68,0,603381,MALE,PhD,prof-specialty,yachting,own-child,0,0,12-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,WV,Hillsdale,3540 Maple St,17,1,YES,0,1,YES,64800,10800,5400,48600,Audi,A3,1999,N
147,33,129872,08-08-2010,OH,100/300,1000,1334.15,6000000,479224,MALE,High School,craft-repair,reading,not-in-family,53300,-49200,24-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,WV,Springfield,3104 Sky Drive,15,1,YES,2,0,YES,53100,10620,5310,37170,Mercedes,C300,1995,Y
62,28,200152,09-03-2003,IL,100/300,1000,988.45,0,430141,FEMALE,Masters,protective-serv,camping,unmarried,0,0,09-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,NY,Northbrook,4981 Weaver St,3,1,?,1,1,YES,60200,6020,6020,48160,Suburu,Forrestor,2004,Y
289,49,933293,03-02-1993,IL,500/1000,2000,1222.48,0,620757,FEMALE,JD,priv-house-serv,golf,unmarried,0,0,18-01-2015,Parked Car,?,Minor Damage,None,WV,Arlington,6676 Tree Lane,16,1,NO,1,1,YES,5330,1230,820,3280,Suburu,Legacy,2001,N
431,54,485664,25-11-2002,IN,500/1000,2000,1155.55,0,615901,FEMALE,MD,craft-repair,bungie-jumping,unmarried,65700,0,21-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NY,Hillsdale,3930 Embaracadero St,4,3,?,2,0,?,62300,12460,6230,43610,Jeep,Wrangler,2007,N
199,37,982871,27-07-1997,IN,250/500,500,1262.08,0,474615,MALE,JD,tech-support,video-games,wife,48500,0,08-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,NC,Columbus,3422 Flute St,4,1,?,0,3,NO,60170,10940,10940,38290,Nissan,Pathfinder,2011,Y
79,26,206213,08-05-1995,IL,100/300,500,1451.62,0,456446,MALE,Associate,tech-support,kayaking,not-in-family,0,-55700,03-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Columbus,4862 Lincoln Hwy,19,1,NO,2,2,?,40000,8000,4000,28000,BMW,M5,2010,N
116,34,616337,30-08-2012,IN,250/500,500,1737.66,0,470577,MALE,Associate,transport-moving,chess,unmarried,0,-24100,01-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,WV,Northbrook,5719 2nd Lane,1,1,?,1,1,?,97080,16180,16180,64720,BMW,X5,2001,Y
37,23,448961,30-04-2006,IL,500/1000,500,1475.93,0,441648,FEMALE,College,prof-specialty,hiking,husband,0,-67400,16-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,SC,Springfield,3221 Solo Ridge,17,3,YES,1,0,NO,51660,5740,5740,40180,Dodge,RAM,2010,N
106,30,790442,13-04-2003,OH,250/500,500,538.17,0,433782,FEMALE,PhD,transport-moving,reading,own-child,49700,-60200,10-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,NC,Arlington,6660 MLK Drive,23,1,NO,2,2,NO,51120,5680,5680,39760,Mercedes,E400,2005,N
269,44,108844,05-12-2007,IL,100/300,2000,1081.08,0,468104,MALE,JD,priv-house-serv,reading,unmarried,36400,-28700,14-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,SC,Springfield,1699 Oak Drive,14,1,YES,0,2,?,56400,11280,11280,33840,Toyota,Highlander,2014,N
265,40,430029,21-08-2006,IL,250/500,1000,1454.43,0,459407,FEMALE,MD,protective-serv,yachting,husband,0,0,21-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,NY,Arlington,4234 Cherokee Lane,17,3,NO,2,3,?,55120,6890,0,48230,Accura,MDX,2002,N
163,33,529112,08-01-1990,IN,100/300,500,1240.47,0,472573,FEMALE,Associate,other-service,polo,husband,35300,0,18-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,NC,Northbend,7476 4th St,11,3,YES,1,1,?,77110,0,14020,63090,Honda,Civic,2014,N
355,47,939631,18-03-1990,OH,500/1000,2000,1273.7,4000000,433473,MALE,College,other-service,kayaking,husband,0,0,10-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,WV,Arlington,8907 Tree Ave,19,3,NO,2,1,NO,62800,6280,6280,50240,Audi,A3,2003,Y
175,34,866931,07-01-2008,IN,500/1000,1000,1123.87,8000000,446326,FEMALE,PhD,protective-serv,dancing,other-relative,0,0,26-02-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Arlington,6619 Flute Ave,5,1,?,2,0,YES,7290,810,810,5670,Volkswagen,Passat,1995,N
192,35,582011,10-03-1997,IL,100/300,1000,1245.89,0,435481,FEMALE,Masters,exec-managerial,movies,wife,0,-40300,01-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,WV,Springfield,6011 Britain St,19,1,NO,0,0,?,76600,15320,7660,53620,Mercedes,C300,2000,N
430,59,691189,10-01-2004,OH,250/500,2000,1326.62,7000000,477310,MALE,MD,other-service,bungie-jumping,own-child,0,0,03-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Riverwood,5104 Francis Drive,19,3,?,0,3,YES,81800,16360,8180,57260,Nissan,Pathfinder,1998,N
91,27,537546,20-08-1994,IL,100/300,2000,1073.83,0,609930,FEMALE,JD,farming-fishing,polo,husband,0,0,17-01-2015,Vehicle Theft,?,Trivial Damage,None,NY,Arlington,2280 4th Ave,4,1,?,1,2,?,7260,1320,660,5280,BMW,M5,2008,N
217,39,394975,02-06-2002,IN,100/300,1000,1530.52,0,603993,MALE,College,armed-forces,basketball,not-in-family,0,0,22-02-2015,Vehicle Theft,?,Minor Damage,None,WV,Northbend,2644 Elm Drive,8,1,?,2,1,YES,4300,430,430,3440,Toyota,Corolla,2000,N
223,40,729634,28-04-1994,IN,100/300,500,1201.41,0,437818,FEMALE,JD,priv-house-serv,movies,husband,88400,-46500,27-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,NC,Columbus,7466 MLK Ridge,7,3,YES,1,0,?,70510,12820,12820,44870,Suburu,Forrestor,1999,N
195,39,282195,17-08-2014,OH,250/500,1000,1393.57,0,478423,MALE,PhD,machine-op-inspct,movies,not-in-family,47600,-39600,27-02-2015,Parked Car,?,Minor Damage,Police,VA,Northbend,5821 2nd St,5,1,NO,0,1,YES,2640,480,480,1680,Ford,F150,2009,N
22,26,420810,11-08-2007,OH,100/300,1000,1276.57,0,467784,MALE,PhD,craft-repair,skydiving,not-in-family,71500,0,06-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Arlington,6723 Best Drive,3,1,YES,1,2,NO,78900,15780,7890,55230,Chevrolet,Silverado,1995,N
439,56,524836,20-11-2008,IN,250/500,500,1082.49,0,606714,FEMALE,PhD,prof-specialty,chess,unmarried,36100,-55000,28-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Columbus,4866 4th Hwy,12,3,?,2,3,?,56430,0,6270,50160,Honda,CRV,2014,N
94,32,307195,18-10-1995,IN,500/1000,1000,1414.74,0,464691,FEMALE,Masters,adm-clerical,hiking,own-child,0,0,22-02-2015,Parked Car,?,Minor Damage,None,VA,Riverwood,5418 Britain Ave,19,1,NO,1,3,NO,2400,300,300,1800,Chevrolet,Silverado,2014,N
11,39,623648,19-05-1993,IL,250/500,2000,1470.06,0,431683,MALE,PhD,other-service,yachting,husband,56600,-45800,07-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Riverwood,4296 Pine Hwy,22,1,YES,0,1,NO,65790,7310,7310,51170,Saab,93,2007,N
151,36,485372,26-02-2005,OH,250/500,2000,870.63,0,431725,FEMALE,MD,adm-clerical,kayaking,own-child,94800,-58500,06-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,VA,Hillsdale,2299 1st St,12,3,NO,1,1,NO,62920,11440,5720,45760,Ford,Escape,2000,N
154,34,598554,14-02-1990,IN,100/300,500,795.23,0,609216,MALE,PhD,machine-op-inspct,base-jumping,other-relative,36900,0,10-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NY,Springfield,6618 Cherokee Drive,15,3,YES,2,1,?,69480,15440,0,54040,Nissan,Maxima,2014,Y
245,44,303987,30-09-1993,IL,500/1000,1000,1168.2,0,452787,MALE,JD,handlers-cleaners,basketball,husband,69100,0,11-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,OH,Springfield,7459 Flute St,23,3,NO,0,3,NO,44280,7380,3690,33210,Honda,Accord,1997,N
119,32,343161,10-06-2014,IL,500/1000,1000,993.51,0,468767,MALE,High School,armed-forces,bungie-jumping,unmarried,0,-49500,12-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,WV,Hillsdale,3567 4th Drive,12,1,NO,0,3,YES,56300,5630,11260,39410,BMW,M5,2011,N
215,42,519312,28-10-2008,OH,500/1000,500,1848.81,0,435489,MALE,JD,transport-moving,video-games,own-child,0,-49000,06-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,WV,Northbend,2457 Washington Ave,20,3,YES,2,2,YES,68520,11420,5710,51390,Suburu,Legacy,2003,Y
295,42,132902,24-04-2007,OH,250/500,2000,1641.73,5000000,450149,MALE,PhD,sales,chess,not-in-family,62400,0,20-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,VA,Riverwood,1269 Flute Drive,16,3,NO,0,0,NO,59130,6570,6570,45990,Ford,Escape,2006,Y
254,39,332867,13-12-1993,IN,100/300,500,1362.87,0,458364,FEMALE,MD,exec-managerial,chess,other-relative,35700,0,22-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Arlington,1218 Sky Hwy,6,3,YES,2,2,NO,82320,13720,6860,61740,Dodge,Neon,1995,Y
107,31,356590,17-08-2011,IN,250/500,500,1239.22,7000000,476458,FEMALE,High School,tech-support,paintball,not-in-family,43400,-91200,30-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,SC,Springfield,9169 Pine Ridge,12,1,YES,0,1,NO,89700,13800,13800,62100,Audi,A5,2009,Y
478,64,346002,20-08-1990,OH,250/500,500,835.02,0,602433,FEMALE,Associate,adm-clerical,reading,unmarried,59600,0,02-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,WV,Hillsdale,8538 Texas Lane,17,3,NO,1,1,NO,33930,0,3770,30160,BMW,X6,1998,N
128,30,500533,11-02-1994,OH,100/300,1000,1061.33,0,478575,MALE,MD,machine-op-inspct,movies,own-child,43300,-66200,10-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,WV,Northbrook,5783 Oak Ave,8,1,NO,0,3,NO,68530,12460,6230,49840,Audi,A5,1997,N
338,49,348209,22-02-1994,IN,500/1000,1000,1279.08,0,449718,MALE,MD,other-service,kayaking,own-child,0,-51500,27-02-2015,Parked Car,?,Minor Damage,None,NC,Riverwood,7721 Washington Ridge,13,1,NO,0,1,?,4300,860,860,2580,Ford,F150,2004,N
271,42,486676,15-08-2011,OH,100/300,500,1105.49,0,463181,FEMALE,Associate,prof-specialty,sleeping,own-child,56200,-50000,20-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,SC,Hillsdale,8006 Maple Hwy,12,2,?,2,3,?,68310,12420,6210,49680,Audi,A3,2003,Y
222,41,260845,11-11-1998,OH,100/300,2000,1055.53,0,441992,FEMALE,MD,armed-forces,cross-fit,not-in-family,37800,-50300,08-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,WV,Northbrook,6751 Pine Ridge,7,1,NO,0,2,NO,61290,6810,6810,47670,Honda,Civic,1995,Y
199,41,657045,04-12-1995,OH,250/500,1000,895.83,0,452597,FEMALE,Associate,sales,paintball,husband,0,0,11-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NC,Arlington,2324 Texas Ridge,10,1,NO,1,2,NO,30100,3010,0,27090,Chevrolet,Malibu,1999,N
215,37,761189,28-12-2002,IN,100/300,500,1632.93,0,614417,FEMALE,College,transport-moving,golf,not-in-family,0,-42900,23-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Riverwood,7923 Elm Ave,7,3,NO,2,0,YES,57120,9520,4760,42840,Mercedes,C300,2002,N
192,40,175177,15-04-2004,IL,100/300,1000,1405.99,0,472895,FEMALE,Associate,sales,yachting,wife,0,0,01-03-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,VA,Springfield,4755 Best Lane,18,3,YES,1,0,YES,42930,9540,4770,28620,BMW,X6,2005,N
120,35,116700,02-02-2001,OH,100/300,1000,1425.54,0,475847,FEMALE,High School,transport-moving,bungie-jumping,other-relative,78300,0,15-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Riverwood,5053 Tree Drive,22,3,NO,2,0,NO,51210,11380,5690,34140,Ford,Fusion,2010,N
270,45,166264,12-01-2010,OH,500/1000,1000,1038.09,0,476978,FEMALE,College,handlers-cleaners,golf,husband,0,-19700,14-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Springfield,2078 3rd Ave,18,3,NO,1,1,YES,89400,14900,7450,67050,Suburu,Legacy,1998,N
319,47,527945,14-04-1992,IN,250/500,500,1307.11,0,600648,MALE,College,transport-moving,dancing,not-in-family,0,0,17-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Northbrook,2804 Best St,22,3,NO,0,2,?,59730,10860,10860,38010,Audi,A3,2005,N
194,39,627540,21-05-2010,OH,500/1000,1000,1489.24,6000000,608335,FEMALE,JD,other-service,kayaking,wife,0,-45000,24-01-2015,Vehicle Theft,?,Minor Damage,None,SC,Springfield,7877 Sky Lane,15,1,YES,2,2,YES,8060,1240,1240,5580,Saab,95,2004,N
227,38,279422,27-10-2013,OH,500/1000,500,976.67,0,471600,FEMALE,PhD,handlers-cleaners,polo,unmarried,0,0,21-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,SC,Northbrook,6530 Weaver Ave,16,1,?,1,2,?,72200,14440,7220,50540,BMW,M5,2013,Y
137,31,484200,12-10-1994,OH,250/500,2000,1340.43,0,441175,MALE,High School,exec-managerial,paintball,husband,52700,-40600,19-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,NC,Arlington,3087 Oak Hwy,6,3,NO,1,2,NO,50800,10160,10160,30480,Accura,MDX,2005,N
244,40,645258,04-07-1997,OH,500/1000,2000,1267.81,5000000,603123,FEMALE,Masters,exec-managerial,video-games,wife,0,0,03-01-2015,Vehicle Theft,?,Trivial Damage,None,NC,Northbrook,7098 Lincoln Hwy,10,1,?,2,1,?,6600,660,1320,4620,Accura,TL,2005,N
78,29,694662,15-02-2011,IL,250/500,1000,1234.2,6000000,457767,MALE,Masters,other-service,bungie-jumping,other-relative,0,0,29-01-2015,Vehicle Theft,?,Minor Damage,Police,NY,Northbrook,5124 Maple St,3,1,YES,2,2,NO,7500,750,1500,5250,Nissan,Maxima,2002,N
200,35,960680,21-08-1994,IN,250/500,2000,1318.06,0,618498,MALE,High School,exec-managerial,video-games,wife,57300,-80600,19-01-2015,Vehicle Theft,?,Trivial Damage,None,VA,Hillsdale,2333 Maple Lane,13,1,NO,0,3,YES,6490,1180,1180,4130,Volkswagen,Jetta,2002,N
284,48,498140,15-05-1997,IN,500/1000,2000,769.95,0,605486,MALE,Masters,prof-specialty,movies,not-in-family,0,-44200,19-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Hillsdale,1012 5th Lane,16,2,?,2,3,NO,60940,5540,11080,44320,Audi,A3,2013,Y
275,41,498875,26-10-1996,OH,100/300,2000,1514.72,0,617970,MALE,High School,transport-moving,board-games,own-child,35700,0,02-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,NY,Northbrook,7477 MLK Drive,13,3,YES,0,1,?,58300,5830,11660,40810,Suburu,Legacy,2007,N
153,34,798177,04-03-2006,IL,500/1000,1000,873.64,4000000,432934,FEMALE,Associate,priv-house-serv,yachting,husband,800,0,30-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,SC,Columbus,9489 3rd St,9,3,NO,2,1,?,68400,11400,11400,45600,Ford,F150,2007,N
134,32,614763,02-01-1991,IL,500/1000,500,1612.43,0,456762,FEMALE,MD,other-service,yachting,own-child,36400,0,08-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,VA,Springfield,2087 Apache Ave,2,1,?,2,1,YES,64240,11680,11680,40880,BMW,3 Series,2015,N
31,36,679370,15-08-1999,IL,500/1000,2000,1318.24,9000000,601748,FEMALE,College,prof-specialty,kayaking,not-in-family,0,-78600,30-01-2015,Parked Car,?,Trivial Damage,None,WV,Arlington,5540 Sky St,9,1,NO,0,1,YES,4700,940,470,3290,Dodge,Neon,2002,N
41,25,958857,15-01-1992,IN,100/300,1000,1226.83,0,607763,FEMALE,College,exec-managerial,exercise,not-in-family,0,-56100,07-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,SC,Columbus,7238 2nd St,12,3,YES,2,0,?,45120,0,5640,39480,Accura,MDX,2011,Y
127,29,686816,07-12-1999,OH,250/500,2000,1326.44,5000000,436973,FEMALE,High School,sales,board-games,own-child,0,0,24-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,SC,Arlington,8442 Britain Hwy,12,2,YES,1,1,?,66950,10300,10300,46350,Saab,93,1995,N
61,23,127754,06-06-1993,IL,250/500,2000,1136.83,4000000,471300,FEMALE,Associate,tech-support,exercise,own-child,0,-62400,02-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,NY,Columbus,1331 Britain Hwy,14,1,NO,0,3,?,98340,8940,17880,71520,Honda,Accord,2004,Y
207,42,918629,03-10-2000,IL,250/500,2000,1322.78,0,453277,MALE,PhD,farming-fishing,yachting,own-child,55200,0,28-02-2015,Parked Car,?,Trivial Damage,None,WV,Springfield,5260 Francis Drive,9,1,NO,0,1,NO,5900,590,590,4720,BMW,X5,2007,N
219,43,731450,29-12-2010,IN,100/300,1000,1483.25,0,465100,FEMALE,MD,exec-managerial,exercise,not-in-family,90700,-20800,09-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NC,Riverwood,1135 Solo Lane,3,3,NO,1,1,?,70680,5890,11780,53010,Ford,Fusion,2009,N
271,42,307447,17-03-1990,IL,100/300,500,1515.3,0,603248,FEMALE,High School,machine-op-inspct,hiking,not-in-family,0,0,19-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Northbend,9737 Solo Hwy,21,3,NO,1,0,NO,93720,17040,8520,68160,Mercedes,ML350,2005,N
80,25,992145,01-03-2012,IL,100/300,2000,1075.18,5000000,601112,FEMALE,PhD,armed-forces,exercise,husband,67700,-58400,21-02-2015,Vehicle Theft,?,Minor Damage,None,OH,Northbrook,3289 Britain Drive,5,1,NO,2,0,YES,6930,1260,630,5040,Toyota,Highlander,2001,N
325,47,900628,05-02-2006,IN,500/1000,1000,1690.27,0,438830,FEMALE,Associate,protective-serv,hiking,not-in-family,61500,0,14-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,VA,Springfield,6550 Andromedia St,11,1,YES,0,3,NO,72930,6630,6630,59670,Dodge,RAM,2006,Y
29,25,235220,01-11-2014,IL,250/500,2000,1352.83,0,464959,MALE,Masters,farming-fishing,skydiving,own-child,0,-71700,22-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,SC,Hillsdale,1679 2nd Hwy,4,4,YES,1,2,YES,64890,7210,7210,50470,Nissan,Pathfinder,2013,Y
295,48,740019,17-06-2009,OH,250/500,1000,1148.73,0,439787,FEMALE,College,machine-op-inspct,kayaking,wife,0,0,22-02-2015,Parked Car,?,Trivial Damage,None,WV,Columbus,3998 Flute St,6,1,?,1,2,YES,5400,900,900,3600,Saab,95,1999,N
239,42,246882,20-09-1999,IL,100/300,1000,969.5,0,464839,MALE,College,exec-managerial,reading,not-in-family,0,0,26-01-2015,Vehicle Theft,?,Trivial Damage,None,NC,Northbrook,2430 MLK Ave,10,1,NO,0,0,?,5600,700,700,4200,Audi,A3,2007,N
269,41,797613,19-10-1990,IN,100/300,500,1463.82,0,448984,FEMALE,College,protective-serv,yachting,not-in-family,0,-72300,24-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Northbend,7717 Britain Hwy,23,1,YES,0,0,?,79300,15860,15860,47580,Saab,92x,2007,N
80,27,193442,05-08-1996,IL,100/300,1000,1474.17,0,440327,FEMALE,College,tech-support,exercise,unmarried,0,0,19-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,WV,Northbend,7773 Tree Hwy,13,1,YES,1,0,YES,52800,10560,5280,36960,Saab,95,2004,N
279,41,389238,06-06-2001,IL,250/500,500,1497.35,0,460742,FEMALE,JD,prof-specialty,bungie-jumping,husband,37300,-31700,29-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NC,Northbrook,2199 Texas Drive,16,3,?,2,3,NO,28800,0,3600,25200,Ford,Fusion,2013,N
165,33,760179,25-03-2007,OH,100/300,1000,1427.14,0,446895,FEMALE,Associate,tech-support,kayaking,other-relative,35300,-58100,15-02-2015,Parked Car,?,Minor Damage,Police,WV,Northbend,1028 Sky Lane,3,1,NO,1,1,NO,2970,330,330,2310,Toyota,Highlander,2008,N
350,54,939905,31-10-2013,OH,500/1000,500,1495.1,0,609374,MALE,College,other-service,basketball,wife,50500,0,12-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Northbend,4154 Lincoln Hwy,15,3,NO,0,0,?,93480,15580,7790,70110,Chevrolet,Malibu,2014,N
295,49,872814,13-06-1992,IL,100/300,500,1141.62,0,451672,MALE,College,prof-specialty,kayaking,husband,34300,-24300,01-01-2015,Vehicle Theft,?,Minor Damage,None,WV,Columbus,8085 Andromedia St,4,1,YES,1,3,YES,4320,480,480,3360,Mercedes,E400,2002,N
464,61,632627,07-10-1990,OH,500/1000,1000,1125.37,0,604450,FEMALE,Associate,prof-specialty,basketball,husband,0,-56400,13-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,VA,Northbend,4793 4th Ridge,6,3,?,0,2,YES,79800,6650,19950,53200,Saab,95,2000,Y
118,28,283414,28-12-1991,IN,500/1000,2000,1207.36,0,432896,FEMALE,High School,handlers-cleaners,camping,own-child,0,-57000,01-03-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,WV,Columbus,7428 Sky Hwy,22,2,NO,1,0,?,74200,7420,14840,51940,Volkswagen,Passat,1997,N
298,47,163161,11-11-1998,IL,500/1000,2000,1338.5,0,618929,FEMALE,PhD,machine-op-inspct,basketball,other-relative,28800,0,02-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,NY,Northbrook,2306 5th Lane,1,1,?,0,0,?,70590,10860,10860,48870,Saab,93,2000,Y
87,31,853360,26-06-2009,IN,500/1000,1000,1074.07,0,451312,FEMALE,Masters,sales,chess,husband,0,-47500,27-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NY,Springfield,3052 Weaver Ridge,16,3,NO,0,3,YES,60940,5540,11080,44320,Nissan,Ultima,2006,Y
261,42,776860,11-01-2009,OH,250/500,500,1337.56,0,605141,FEMALE,College,prof-specialty,video-games,unmarried,0,0,12-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Riverwood,5211 Weaver Drive,18,1,?,1,2,YES,74700,7470,14940,52290,Dodge,RAM,2010,N
453,60,149367,18-03-2003,IN,100/300,500,1298.91,6000000,459504,MALE,PhD,craft-repair,dancing,unmarried,52600,-38800,06-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,NC,Springfield,7253 MLK St,0,1,YES,0,0,?,70000,14000,7000,49000,Ford,F150,2015,Y
210,41,395269,02-11-2012,IL,500/1000,500,1222.75,0,432781,MALE,High School,exec-managerial,polo,other-relative,0,-41000,30-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,WV,Columbus,1454 5th Ridge,12,3,?,2,0,NO,81070,14740,14740,51590,BMW,X5,2001,N
168,32,981123,04-05-2000,IN,100/300,1000,1059.52,0,452748,MALE,MD,protective-serv,camping,own-child,0,-40600,01-03-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,VA,Hillsdale,5622 Best Ridge,13,2,?,1,1,?,57720,14430,9620,33670,Saab,93,2007,N
390,51,143626,29-09-1999,OH,250/500,2000,1124.38,0,618316,MALE,Associate,armed-forces,reading,other-relative,0,0,24-02-2015,Vehicle Theft,?,Minor Damage,None,VA,Northbrook,4574 Britain Hwy,9,1,YES,1,1,YES,7080,1180,1180,4720,Ford,F150,2001,N
258,46,648397,09-03-1999,IN,100/300,1000,1110.37,10000000,455365,MALE,MD,machine-op-inspct,hiking,other-relative,34400,-56800,24-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,NY,Riverwood,4539 Texas St,14,3,NO,0,1,?,47700,4770,9540,33390,Accura,MDX,1997,Y
107,31,154982,13-02-1991,IL,500/1000,2000,1374.22,0,470603,FEMALE,PhD,machine-op-inspct,bungie-jumping,other-relative,62000,-63100,26-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Hillsdale,8118 Elm Ridge,16,1,NO,0,3,NO,51260,9320,9320,32620,Suburu,Legacy,2002,N
225,41,330591,05-08-1993,OH,500/1000,2000,1103.58,0,475292,MALE,High School,exec-managerial,exercise,unmarried,41200,-36200,19-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,NY,Northbend,3814 Britain Drive,20,3,?,2,2,NO,70400,6400,12800,51200,Toyota,Highlander,2011,Y
164,38,319232,31-10-1997,IL,250/500,2000,1269.76,0,467743,FEMALE,PhD,transport-moving,yachting,not-in-family,44300,0,17-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Hillsdale,4614 MLK Ave,4,1,NO,1,3,YES,90000,18000,9000,63000,Ford,Fusion,2015,N
245,39,531640,21-04-2001,OH,250/500,500,964.79,8000000,460675,FEMALE,Associate,adm-clerical,camping,husband,58000,0,20-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Arlington,1628 Best Drive,7,3,?,0,1,?,72820,13240,6620,52960,BMW,3 Series,2010,N
255,41,368050,08-01-2013,IL,500/1000,2000,1167.3,4000000,618123,MALE,High School,priv-house-serv,board-games,other-relative,0,0,22-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,SC,Hillsdale,8381 Solo Hwy,22,1,NO,2,0,?,69300,13860,13860,41580,Volkswagen,Passat,2000,N
206,36,253791,23-07-2009,IL,500/1000,500,1625.45,4000000,607452,FEMALE,MD,other-service,video-games,other-relative,0,-53700,23-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Northbrook,2100 MLK St,11,1,NO,2,1,NO,76560,12760,12760,51040,Ford,Fusion,2008,Y
203,38,155724,20-02-1998,IL,250/500,500,1394.43,0,606352,FEMALE,Masters,other-service,skydiving,not-in-family,0,0,31-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,VA,Columbus,5071 Flute Ridge,7,1,?,0,1,YES,55440,0,6160,49280,Nissan,Maxima,1999,Y
22,25,824540,13-03-2008,OH,250/500,2000,1053.24,0,603527,FEMALE,College,prof-specialty,movies,other-relative,51100,0,05-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,NC,Arlington,7551 Britain Lane,0,4,YES,1,0,NO,77130,8570,17140,51420,Accura,MDX,1995,N
211,35,717392,20-08-1996,IL,100/300,500,1040.75,0,445601,FEMALE,JD,prof-specialty,paintball,not-in-family,0,0,03-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Northbend,2275 Best Lane,1,1,YES,1,1,?,42000,7000,7000,28000,BMW,X5,2011,N
206,39,965768,27-07-2014,IN,250/500,1000,1302.4,6000000,603948,MALE,JD,craft-repair,dancing,unmarried,47200,-69700,17-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,NY,Hillsdale,1598 3rd Drive,12,3,NO,2,3,YES,36300,3300,9900,23100,Ford,Escape,2013,N
166,38,414779,09-11-1992,IL,100/300,2000,1588.55,0,435758,MALE,MD,protective-serv,video-games,unmarried,59600,-32100,09-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,WV,Columbus,7740 MLK St,8,1,NO,0,0,?,40320,5760,5760,28800,Suburu,Impreza,2001,N
165,32,428230,04-06-2012,IN,500/1000,500,1399.26,0,611586,FEMALE,High School,tech-support,exercise,own-child,70500,0,08-02-2015,Parked Car,?,Minor Damage,Police,PA,Northbrook,1240 Tree Lane,9,1,?,2,0,?,3960,330,660,2970,BMW,M5,1998,N
274,43,517240,13-05-2001,OH,100/300,2000,1352.31,0,465263,MALE,College,farming-fishing,basketball,wife,40700,-47300,06-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,NY,Northbend,8983 Francis Ridge,22,1,YES,0,3,YES,63840,10640,10640,42560,BMW,X5,2006,Y
81,28,469874,17-09-2011,IL,250/500,1000,1139,6000000,617858,FEMALE,Masters,sales,sleeping,husband,42400,0,14-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Northbend,7756 Solo Drive,0,3,?,1,2,?,44730,4970,4970,34790,Saab,92x,2000,Y
280,45,718428,15-07-2011,IN,250/500,1000,1397.67,0,607889,MALE,JD,machine-op-inspct,camping,wife,57900,0,22-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,NY,Arlington,9034 Weaver Ridge,0,3,YES,1,2,NO,84720,14120,14120,56480,Ford,Escape,1999,N
194,39,620215,27-07-2005,IN,250/500,500,823.17,0,455689,MALE,Masters,adm-clerical,paintball,own-child,0,0,23-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,WV,Columbus,1126 Texas Hwy,3,3,?,2,2,NO,61500,6150,12300,43050,Dodge,RAM,2012,N
112,27,618659,18-10-2005,OH,100/300,500,965.13,0,450341,FEMALE,Masters,tech-support,exercise,unmarried,60000,-54800,22-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NC,Arlington,2808 Elm St,21,3,?,0,1,?,51000,8500,8500,34000,Nissan,Pathfinder,2013,N
24,33,649082,19-01-1996,IL,500/1000,1000,1922.84,0,431277,FEMALE,High School,machine-op-inspct,skydiving,wife,0,-45200,24-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,WV,Northbend,5061 Francis Ave,0,1,?,2,1,NO,46800,4680,9360,32760,Jeep,Wrangler,2002,N
93,32,437573,29-09-2005,OH,250/500,2000,1624.82,0,454656,MALE,PhD,exec-managerial,basketball,unmarried,65300,-65600,09-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Northbend,4965 MLK Drive,16,1,?,1,1,?,78120,17360,8680,52080,BMW,3 Series,2006,N
171,34,964657,18-02-1997,IN,250/500,2000,1277.25,0,605169,FEMALE,College,exec-managerial,yachting,other-relative,84900,0,19-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,NY,Springfield,8668 Flute St,14,1,NO,1,2,?,69200,13840,6920,48440,Volkswagen,Passat,1996,Y
200,40,932502,11-05-2010,IL,100/300,1000,1439.34,0,444822,FEMALE,High School,sales,exercise,other-relative,45300,-20400,01-01-2015,Vehicle Theft,?,Minor Damage,Police,VA,Riverwood,2577 Washington Drive,9,1,?,0,0,NO,3690,410,410,2870,Ford,Escape,2015,N
120,28,434507,06-02-2009,IL,250/500,1000,1281.27,0,447442,FEMALE,PhD,tech-support,golf,not-in-family,68900,0,07-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Springfield,7709 Rock Lane,9,3,NO,0,3,?,65500,6550,6550,52400,BMW,X5,2010,N
325,46,935277,09-07-2013,IL,500/1000,500,1348.83,0,474360,FEMALE,High School,prof-specialty,basketball,wife,46300,-77500,01-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,NC,Springfield,9358 Texas Ridge,21,3,?,1,2,YES,76120,13840,6920,55360,Toyota,Camry,1999,N
124,32,756054,06-06-1992,IL,250/500,1000,1198.15,0,447925,FEMALE,MD,other-service,hiking,not-in-family,0,-43200,21-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,VA,Springfield,8080 Oak Lane,19,3,NO,0,2,YES,73560,12260,12260,49040,BMW,X5,1995,N
211,35,682387,08-03-1998,OH,100/300,2000,1221.22,0,451586,MALE,Masters,machine-op-inspct,camping,other-relative,76000,0,21-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,NY,Northbend,6408 Weaver Ridge,2,1,YES,0,1,?,52030,9460,4730,37840,Mercedes,E400,2008,N
287,41,456604,29-03-2004,IL,500/1000,2000,968.74,0,477519,MALE,Masters,transport-moving,video-games,wife,0,-49000,19-01-2015,Vehicle Theft,?,Trivial Damage,None,SC,Northbrook,5532 Weaver Ridge,9,1,?,2,3,?,5170,470,940,3760,Suburu,Forrestor,2001,N
122,34,139872,01-06-2006,IN,250/500,1000,1220.71,0,603639,MALE,PhD,machine-op-inspct,video-games,own-child,58600,-28700,27-02-2015,Parked Car,?,Minor Damage,None,SC,Northbend,9101 2nd Hwy,5,1,?,2,1,YES,8190,1890,1260,5040,Chevrolet,Silverado,2013,N
22,29,354105,08-06-1994,IN,250/500,2000,1238.62,6000000,463993,MALE,MD,exec-managerial,yachting,other-relative,0,-56200,14-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Springfield,8576 Andromedia St,14,1,?,2,3,?,70800,7080,14160,49560,Accura,MDX,2012,Y
106,31,165485,12-02-1998,IL,500/1000,2000,1320.75,0,441491,FEMALE,JD,farming-fishing,video-games,wife,54100,0,01-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,WV,Springfield,6315 2nd Lane,20,3,?,0,3,YES,45630,5070,5070,35490,Accura,MDX,2011,N
398,58,515050,16-11-2000,OH,100/300,500,990.98,0,469429,FEMALE,Associate,exec-managerial,cross-fit,wife,0,-57900,02-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NC,Northbrook,1536 Flute Drive,18,1,YES,2,1,?,99320,7640,15280,76400,Accura,TL,2002,Y
214,41,795686,24-10-2004,IL,500/1000,500,1398.51,4000000,472214,MALE,Masters,tech-support,polo,not-in-family,0,-57100,01-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,NC,Northbend,4672 MLK St,13,3,?,2,0,NO,64000,12800,6400,44800,BMW,X6,1996,Y
209,38,395983,08-11-2009,OH,100/300,500,1355.08,0,614945,FEMALE,MD,other-service,golf,other-relative,58100,0,19-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbrook,2204 Washington Lane,21,1,?,1,1,?,47300,4730,4730,37840,Nissan,Pathfinder,2014,N
82,27,119513,21-09-1996,IL,100/300,1000,1384.51,0,476727,MALE,PhD,adm-clerical,reading,not-in-family,13100,-38200,18-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,SC,Springfield,9484 Pine Drive,14,1,NO,2,3,?,71680,8960,8960,53760,Volkswagen,Jetta,2007,Y
193,41,217938,16-07-1995,OH,250/500,500,847.03,0,438555,FEMALE,JD,craft-repair,skydiving,not-in-family,0,0,08-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,SC,Springfield,5431 3rd Ridge,1,1,?,1,0,YES,112320,17280,17280,77760,Suburu,Impreza,2011,Y
134,32,203914,09-06-2001,OH,100/300,1000,1000.06,0,440961,FEMALE,PhD,farming-fishing,base-jumping,wife,0,0,09-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Columbus,7121 Britain Drive,17,1,?,1,0,?,82720,7520,15040,60160,Audi,A3,2014,N
288,45,565157,06-10-2002,IL,100/300,1000,1046.71,0,616714,MALE,Masters,priv-house-serv,polo,husband,0,0,27-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NC,Hillsdale,8586 1st Ridge,4,3,?,2,1,?,48060,10680,5340,32040,Mercedes,C300,2009,N
104,32,904191,14-07-1997,IN,250/500,500,1158.03,0,434247,MALE,High School,exec-managerial,kayaking,own-child,31900,-44600,08-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,NY,Riverwood,7582 Pine Drive,23,3,YES,1,3,YES,63570,9780,9780,44010,Volkswagen,Jetta,2006,Y
431,54,419510,11-11-1994,OH,100/300,1000,1372.27,0,436547,FEMALE,Masters,craft-repair,paintball,own-child,17600,0,08-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,WV,Columbus,1388 Embaracadero Hwy,13,3,NO,2,3,?,63240,10540,10540,42160,Suburu,Forrestor,1997,N
101,33,575000,23-06-2012,OH,100/300,1000,1053.04,7000000,619540,FEMALE,Masters,other-service,reading,own-child,52000,-44500,24-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,NC,Northbrook,5621 4th Ave,20,3,NO,1,3,?,54240,9040,9040,36160,Saab,93,2013,Y
375,50,120485,18-02-2007,OH,100/300,1000,1275.39,0,466283,MALE,Associate,sales,bungie-jumping,other-relative,0,0,12-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NY,Riverwood,8150 Washington Ridge,16,1,YES,2,3,NO,37280,0,0,37280,Audi,A5,1996,Y
461,61,781181,27-06-2005,OH,100/300,2000,1402.75,0,449557,MALE,JD,exec-managerial,exercise,husband,0,0,18-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Columbus,4268 2nd Ave,10,3,?,2,1,YES,72100,7210,14420,50470,Jeep,Wrangler,2006,N
428,59,299796,29-09-1999,IN,250/500,500,1344.36,7000000,473329,FEMALE,JD,prof-specialty,hiking,other-relative,0,0,06-02-2015,Parked Car,?,Minor Damage,None,WV,Northbend,6375 2nd Lane,8,1,?,2,3,NO,6500,1300,650,4550,Saab,92x,2013,N
45,38,589749,14-05-2006,IN,100/300,1000,1197.71,0,470117,MALE,Masters,machine-op-inspct,movies,not-in-family,29000,0,14-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Springfield,3770 Flute Drive,17,3,?,2,0,?,78240,13040,13040,52160,Suburu,Legacy,2013,N
136,29,854021,29-04-2010,OH,100/300,500,1203.24,0,600702,FEMALE,JD,transport-moving,video-games,other-relative,62500,-66900,05-02-2015,Vehicle Theft,?,Minor Damage,Police,NY,Columbus,1562 Britain St,9,1,?,2,0,?,6200,1240,620,4340,Honda,Accord,1999,N
216,36,454086,10-11-1992,IN,500/1000,1000,1152.4,0,615921,FEMALE,Associate,priv-house-serv,reading,unmarried,39600,-82400,25-01-2015,Parked Car,?,Minor Damage,Police,VA,Springfield,1681 Cherokee Hwy,0,1,YES,2,3,?,6160,560,1680,3920,Mercedes,E400,2014,N
278,48,139484,24-07-1999,IN,500/1000,2000,1142.62,7000000,475588,FEMALE,MD,farming-fishing,dancing,not-in-family,0,-54000,16-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Northbrook,7523 Oak Lane,12,1,?,2,0,?,76050,11700,11700,52650,Chevrolet,Silverado,1997,N
295,48,678849,22-02-1992,OH,500/1000,1000,1332.07,0,609409,FEMALE,MD,exec-managerial,polo,unmarried,49700,-59100,28-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,NY,Hillsdale,1815 Cherokee Drive,20,2,NO,2,1,YES,86060,13240,13240,59580,Volkswagen,Passat,2002,N
112,30,346940,13-09-2002,OH,500/1000,1000,1166.54,0,479852,FEMALE,Masters,prof-specialty,sleeping,not-in-family,47700,-59300,21-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Arlington,9316 Pine Ave,3,1,YES,2,0,NO,107900,10790,21580,75530,Dodge,Neon,1997,Y
122,34,985436,09-08-2003,IL,250/500,500,1495.06,0,452249,FEMALE,Masters,prof-specialty,polo,unmarried,38100,-31400,07-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,PA,Columbus,2733 Texas Drive,18,3,NO,2,0,YES,99990,18180,18180,63630,Mercedes,E400,2011,N
108,29,237418,04-12-2007,IN,500/1000,1000,1337.92,0,441536,FEMALE,PhD,armed-forces,bungie-jumping,not-in-family,71400,0,23-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Columbus,7684 Francis Ridge,4,3,NO,2,2,YES,61380,11160,5580,44640,Suburu,Legacy,2012,N
14,28,335780,22-07-2002,OH,250/500,2000,1587.96,0,601617,FEMALE,Associate,craft-repair,board-games,unmarried,0,-26900,14-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,WV,Riverwood,8991 Embaracadero Ave,18,1,?,2,1,YES,71280,12960,12960,45360,Audi,A5,2012,Y
298,45,491392,03-07-1992,IL,500/1000,1000,1362.29,0,442598,MALE,High School,farming-fishing,yachting,unmarried,0,-51100,27-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Northbend,4905 Francis Ave,18,3,NO,0,0,NO,64000,12800,6400,44800,Saab,95,1999,N
276,46,140880,29-03-2005,IL,250/500,500,1448.84,0,430987,FEMALE,Masters,machine-op-inspct,bungie-jumping,husband,0,-50000,21-02-2015,Parked Car,?,Trivial Damage,Police,SC,Hillsdale,7783 Lincoln Hwy,7,1,?,0,1,NO,5940,660,660,4620,Toyota,Highlander,2015,N
47,37,962591,16-03-2008,IN,250/500,2000,1241.97,0,430104,MALE,High School,other-service,movies,not-in-family,75400,0,05-01-2015,Parked Car,?,Minor Damage,None,NY,Hillsdale,8749 Tree St,18,1,NO,1,0,NO,6700,670,670,5360,Jeep,Wrangler,2011,N
222,42,922565,23-05-1999,IL,250/500,500,1124.6,0,612904,MALE,Associate,armed-forces,hiking,not-in-family,0,0,30-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Arlington,4985 Sky Lane,23,3,YES,0,0,?,51740,11940,7960,31840,Jeep,Wrangler,2006,N
119,28,288580,22-11-2012,OH,250/500,2000,1079.92,0,430886,MALE,High School,machine-op-inspct,hiking,husband,88800,0,26-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,PA,Riverwood,7534 MLK Hwy,1,1,?,0,1,YES,53600,6700,6700,40200,Volkswagen,Jetta,2007,Y
73,29,154280,29-01-1993,IL,250/500,1000,1447.78,0,467947,MALE,College,protective-serv,board-games,wife,35100,-59900,10-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,SC,Arlington,8689 Maple Hwy,15,3,YES,1,0,?,44910,4990,4990,34930,Dodge,Neon,2001,N
8,31,425973,11-02-2003,IN,250/500,500,1229.16,4000000,604804,FEMALE,MD,transport-moving,kayaking,wife,0,-88300,23-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,PA,Hillsdale,9153 3rd Hwy,2,3,?,0,2,YES,48100,11100,7400,29600,Volkswagen,Jetta,2014,N
294,44,477177,15-08-1990,IL,100/300,1000,1226.49,0,460308,FEMALE,PhD,farming-fishing,kayaking,unmarried,53900,0,05-02-2015,Vehicle Theft,?,Trivial Damage,None,WV,Riverwood,5904 1st Drive,3,1,NO,0,1,NO,6100,610,1220,4270,Ford,Fusion,2002,N
324,46,648509,06-03-2010,IN,100/300,2000,897.89,6000000,618862,MALE,MD,tech-support,board-games,wife,0,-41300,21-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,PA,Hillsdale,4519 Embaracadero St,13,1,?,1,0,YES,79600,15920,15920,47760,Jeep,Wrangler,2011,N
155,34,914815,27-09-1990,IN,100/300,500,1706.79,0,462479,MALE,Masters,protective-serv,dancing,other-relative,0,0,07-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,OH,Hillsdale,9706 MLK Lane,1,1,NO,1,1,YES,77040,8560,8560,59920,Honda,Civic,1998,N
261,45,249048,17-06-2005,IL,250/500,1000,1254.18,0,457555,FEMALE,PhD,prof-specialty,kayaking,other-relative,0,-45100,11-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Columbus,6012 Texas Hwy,16,1,YES,0,1,?,62590,11380,11380,39830,Volkswagen,Jetta,2003,N
245,40,144323,14-09-2001,IN,500/1000,500,885.08,0,459984,FEMALE,Masters,armed-forces,skydiving,other-relative,27000,-58900,06-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,WV,Northbend,4098 Weaver Ridge,0,1,YES,0,1,NO,85150,13100,13100,58950,Chevrolet,Malibu,1998,N
235,39,651861,07-01-2011,IL,100/300,500,1046.58,4000000,434982,MALE,MD,tech-support,exercise,wife,0,-31700,24-01-2015,Vehicle Theft,?,Trivial Damage,None,NY,Hillsdale,6193 1st Hwy,1,1,?,2,1,NO,4950,450,900,3600,Chevrolet,Silverado,2010,N
53,36,125324,13-09-2003,OH,500/1000,2000,1712.68,0,614233,MALE,Associate,handlers-cleaners,basketball,not-in-family,72200,0,18-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,VA,Riverwood,4053 Sky Lane,17,3,?,2,0,YES,51100,10220,5110,35770,Audi,A3,2006,N
426,54,398102,24-10-1997,IL,500/1000,2000,1097.71,0,605258,FEMALE,Masters,adm-clerical,reading,not-in-family,29600,-22300,12-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Springfield,8964 Francis St,13,1,YES,2,3,?,100800,16800,16800,67200,Dodge,Neon,1997,N
111,27,514065,04-01-2009,IN,250/500,500,1363.77,4000000,604377,FEMALE,Masters,tech-support,exercise,husband,51100,0,17-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,VA,Springfield,9748 Sky Drive,12,1,YES,2,3,NO,90970,16540,16540,57890,Accura,RSX,2009,N
86,26,391652,12-10-1998,OH,100/300,500,1382.88,7000000,434923,MALE,JD,tech-support,cross-fit,other-relative,0,-30300,12-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Northbend,2293 Washington Ave,11,1,?,1,1,YES,81840,14880,7440,59520,Chevrolet,Malibu,2011,Y
296,46,922167,23-02-1993,OH,100/300,1000,1141.35,7000000,476456,MALE,Masters,craft-repair,sleeping,not-in-family,0,0,06-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Northbend,3656 Solo Ave,18,1,?,0,2,NO,54900,5490,5490,43920,Mercedes,C300,2013,N
125,35,442795,07-07-1996,OH,500/1000,500,1054.83,7000000,446788,MALE,JD,tech-support,cross-fit,husband,0,-51300,25-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Northbend,8579 Apache Drive,17,1,YES,2,3,NO,88660,8060,16120,64480,Mercedes,C300,2007,Y
177,34,226330,23-01-2013,IL,100/300,2000,1057.77,0,477382,FEMALE,JD,tech-support,bungie-jumping,unmarried,0,-57700,16-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,NC,Riverwood,2003 Maple Hwy,22,3,?,1,1,NO,18000,2250,2250,13500,Audi,A3,2009,N
238,39,134430,06-12-2006,IN,250/500,2000,1488.02,0,600275,FEMALE,JD,protective-serv,exercise,other-relative,0,-39200,25-02-2015,Vehicle Theft,?,Minor Damage,Police,NY,Northbrook,5445 Tree Hwy,9,1,YES,1,3,YES,5500,1100,550,3850,Chevrolet,Tahoe,2010,N
81,25,524230,23-02-2014,IN,100/300,500,920.3,5000000,461958,FEMALE,High School,tech-support,hiking,own-child,51000,-67900,21-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,WV,Hillsdale,9730 2nd Hwy,11,3,?,1,0,?,73920,13440,6720,53760,Honda,Civic,2003,Y
128,28,438817,16-11-2007,OH,500/1000,1000,986.53,0,472720,FEMALE,High School,adm-clerical,polo,other-relative,62700,0,16-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,VA,Columbus,7819 Oak St,11,1,NO,0,2,YES,101860,18520,18520,64820,Honda,CRV,2007,N
449,57,293794,17-04-1999,OH,250/500,2000,1440.68,0,442395,MALE,Associate,tech-support,movies,own-child,25000,0,09-02-2015,Parked Car,?,Minor Damage,None,SC,Riverwood,1845 Best St,8,1,YES,0,1,YES,5390,980,980,3430,Volkswagen,Passat,2008,N
252,39,868283,06-02-2006,IN,250/500,1000,1086.21,0,455340,MALE,JD,farming-fishing,hiking,unmarried,68500,-57500,13-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Springfield,2500 Tree St,18,1,?,1,2,?,50490,5610,5610,39270,Nissan,Maxima,2001,N
359,47,828890,20-10-1993,OH,100/300,2000,1367.68,0,613247,FEMALE,MD,handlers-cleaners,basketball,unmarried,0,0,11-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,SC,Arlington,6955 Pine Drive,13,1,NO,0,3,NO,55500,5550,11100,38850,Mercedes,C300,2012,N
19,32,882920,01-01-2006,OH,500/1000,1000,1215.85,0,454985,MALE,High School,other-service,hiking,husband,42900,-90200,02-01-2015,Vehicle Theft,?,Minor Damage,Police,VA,Hillsdale,6165 Rock Ridge,8,1,YES,1,1,YES,7040,1280,640,5120,Nissan,Maxima,2015,N
73,26,918777,04-04-2003,IL,250/500,2000,1191.19,4000000,468813,MALE,MD,farming-fishing,basketball,not-in-family,29300,0,12-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,VA,Northbrook,3653 Elm Drive,9,3,?,0,1,YES,40160,5020,0,35140,Chevrolet,Tahoe,2003,N
285,44,212580,05-07-2014,IL,500/1000,1000,1594.45,0,452747,MALE,High School,handlers-cleaners,bungie-jumping,husband,0,0,05-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,NC,Columbus,5812 3rd Hwy,17,2,YES,2,3,NO,55680,6960,6960,41760,Saab,95,2006,N
196,36,602410,16-01-1996,IN,250/500,2000,1463.07,0,615611,MALE,MD,armed-forces,skydiving,own-child,0,0,24-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Springfield,4939 Best St,3,1,?,1,1,NO,5300,530,530,4240,Jeep,Grand Cherokee,2001,Y
223,43,976971,19-04-2002,OH,250/500,500,1734.09,0,451400,FEMALE,MD,adm-clerical,camping,not-in-family,0,0,12-01-2015,Parked Car,?,Trivial Damage,None,WV,Arlington,4964 Elm Lane,6,1,YES,0,3,?,5200,650,650,3900,Accura,MDX,2006,N
328,48,630226,10-12-2005,IL,250/500,500,1411.43,0,464874,MALE,Masters,armed-forces,bungie-jumping,own-child,45100,-32800,16-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Riverwood,9588 Solo St,17,1,YES,2,1,NO,59400,5940,11880,41580,Honda,Civic,2014,N
285,43,171254,07-11-1994,OH,100/300,2000,1512.58,0,452496,FEMALE,College,sales,paintball,other-relative,47600,0,13-01-2015,Vehicle Theft,?,Minor Damage,Police,SC,Arlington,8718 Apache Lane,7,1,?,1,1,?,2520,280,280,1960,BMW,3 Series,1997,N
30,31,247116,02-06-2012,IL,250/500,2000,1153.35,0,430714,MALE,PhD,craft-repair,golf,not-in-family,0,0,02-02-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Hillsdale,3590 Best Hwy,9,1,NO,2,1,YES,5760,960,960,3840,Suburu,Impreza,2011,N
342,49,505969,07-04-1998,OH,250/500,500,1722.95,0,472634,MALE,PhD,transport-moving,base-jumping,not-in-family,63100,-13800,28-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,WV,Hillsdale,6149 Best Ridge,3,1,?,2,0,YES,76700,7670,7670,61360,Suburu,Legacy,2006,N
219,39,653864,25-04-2007,IN,250/500,2000,1281.07,7000000,608371,FEMALE,High School,protective-serv,board-games,unmarried,0,0,18-01-2015,Parked Car,?,Trivial Damage,Police,NY,Springfield,4116 Embaracadero Lane,6,1,NO,0,2,NO,5920,740,740,4440,Chevrolet,Malibu,2015,N
468,62,586367,30-06-2000,IL,100/300,500,1011.92,0,468168,MALE,PhD,machine-op-inspct,paintball,wife,0,0,15-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Riverwood,3486 Flute Ave,7,3,YES,0,3,?,64350,7150,7150,50050,Chevrolet,Tahoe,2009,N
241,39,896890,04-06-1996,IL,250/500,2000,1042.26,0,464107,MALE,JD,sales,kayaking,husband,0,0,31-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Northbend,5994 5th Ave,21,3,NO,1,2,?,19080,4240,2120,12720,Saab,93,1995,N
223,43,650026,09-05-2009,OH,500/1000,500,1235.1,0,466959,FEMALE,Masters,tech-support,exercise,not-in-family,66400,-34400,10-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Springfield,9138 3rd St,14,1,NO,2,1,NO,54400,5440,5440,43520,BMW,M5,2011,N
128,32,547744,08-07-2001,OH,100/300,2000,768.91,0,443522,FEMALE,College,sales,chess,other-relative,0,-39300,26-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,SC,Columbus,3743 Andromedia Ridge,0,1,?,1,0,NO,59800,5980,5980,47840,Ford,F150,1999,Y
124,29,598124,20-09-1993,OH,500/1000,500,1301.72,0,441726,MALE,Masters,handlers-cleaners,golf,husband,0,0,25-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,VA,Springfield,7644 Tree Ridge,14,3,?,0,3,YES,72000,7200,7200,57600,Audi,A3,2005,N
343,48,436126,03-11-2009,IN,250/500,500,1451.54,3000000,473412,MALE,JD,adm-clerical,hiking,husband,0,0,08-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Riverwood,3167 2nd St,13,4,NO,2,3,NO,65070,7230,14460,43380,Mercedes,C300,2003,N
404,53,739447,10-12-2014,IN,250/500,500,767.14,0,466201,MALE,Associate,sales,reading,not-in-family,25500,-36700,14-01-2015,Parked Car,?,Trivial Damage,Police,WV,Columbus,3327 Lincoln Drive,8,1,NO,0,1,NO,8800,1760,880,6160,Suburu,Legacy,2002,N
63,24,427484,08-01-1994,OH,250/500,2000,1620.89,0,469621,FEMALE,High School,handlers-cleaners,movies,other-relative,0,0,03-02-2015,Vehicle Theft,?,Minor Damage,Police,NC,Hillsdale,8621 Best Ridge,7,1,NO,2,0,NO,6120,1020,1020,4080,Toyota,Corolla,2015,N
210,37,218684,05-08-2006,IN,500/1000,2000,1048.46,0,466676,MALE,High School,priv-house-serv,skydiving,not-in-family,59900,0,05-01-2015,Vehicle Theft,?,Trivial Damage,None,OH,Columbus,3878 Tree Lane,9,1,?,1,2,?,7080,1180,590,5310,Dodge,RAM,1999,N
335,50,565564,07-02-2007,OH,100/300,1000,1538.26,6000000,615346,MALE,High School,sales,yachting,other-relative,62200,-31400,24-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Riverwood,9760 Solo Lane,12,3,YES,2,3,YES,34320,8580,4290,21450,Volkswagen,Passat,2009,N
11,40,743163,09-04-2001,OH,500/1000,2000,1217.69,0,440106,FEMALE,MD,prof-specialty,reading,wife,24000,0,26-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,SC,Springfield,9138 1st St,6,3,?,1,0,?,53460,9720,9720,34020,Dodge,Neon,2004,Y
142,33,604614,17-02-1995,IN,100/300,2000,1362.64,5000000,450332,FEMALE,JD,exec-managerial,cross-fit,wife,0,0,21-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NC,Columbus,3414 Elm Ave,11,1,YES,1,3,?,81360,6780,13560,61020,Suburu,Legacy,2009,Y
272,43,509928,25-07-1995,OH,100/300,1000,1279.13,0,615226,MALE,PhD,craft-repair,bungie-jumping,other-relative,0,0,06-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,NC,Springfield,3172 Tree Ridge,5,3,?,1,2,YES,81070,7370,14740,58960,Audi,A3,2006,Y
69,26,593390,24-03-2006,IL,100/300,2000,924.72,0,437688,FEMALE,High School,machine-op-inspct,base-jumping,unmarried,0,0,07-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NY,Columbus,6104 Oak Ave,14,1,NO,2,2,NO,63120,5260,10520,47340,Audi,A5,2008,N
38,28,970607,28-03-1995,OH,250/500,1000,1019.44,0,437387,MALE,Masters,transport-moving,yachting,not-in-family,0,-39700,10-02-2015,Vehicle Theft,?,Trivial Damage,None,NC,Springfield,9742 5th Ridge,17,1,?,2,1,NO,7200,1440,720,5040,BMW,X5,2004,N
328,46,174701,19-06-1996,IL,500/1000,500,1314.6,0,458139,FEMALE,MD,prof-specialty,exercise,not-in-family,24800,0,23-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,WV,Hillsdale,8782 3rd St,0,1,?,2,3,?,70290,12780,6390,51120,Saab,92x,1998,Y
281,43,529398,16-06-1993,OH,100/300,1000,1515.18,6000000,443191,MALE,College,priv-house-serv,camping,other-relative,0,0,09-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,SC,Northbrook,9798 Sky Ridge,21,3,NO,1,0,NO,60190,9260,9260,41670,BMW,X6,1999,N
246,44,940942,11-07-2001,OH,250/500,2000,1649.18,0,613647,MALE,College,farming-fishing,base-jumping,other-relative,0,-58600,22-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Hillsdale,5483 Francis Drive,18,1,YES,1,2,YES,61380,11160,5580,44640,Honda,Civic,2009,Y
298,49,442677,22-11-2008,OH,250/500,500,1451.01,0,460820,FEMALE,College,other-service,exercise,own-child,47800,0,21-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Springfield,2005 Texas Hwy,17,1,NO,2,2,NO,28100,2810,5620,19670,Jeep,Grand Cherokee,2012,N
330,50,365364,28-12-2002,IL,500/1000,1000,978.46,0,431121,FEMALE,High School,sales,yachting,husband,0,0,04-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,NY,Hillsdale,6634 Texas Ridge,19,1,YES,0,0,NO,49060,8920,8920,31220,Chevrolet,Silverado,1995,N
362,50,114839,01-01-2006,IL,250/500,500,1198.34,4000000,619735,MALE,Associate,sales,board-games,wife,53000,-72500,07-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,NY,Columbus,8655 Cherokee Lane,17,3,?,1,1,NO,57060,6340,6340,44380,Mercedes,E400,1995,N
241,38,872734,19-05-1990,IN,100/300,2000,1003.23,0,470485,FEMALE,Associate,tech-support,kayaking,not-in-family,0,0,17-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,VA,Arlington,4955 Lincoln Ridge,23,3,?,0,3,YES,77880,12980,12980,51920,Accura,MDX,2008,N
245,41,267885,26-08-2013,IN,500/1000,2000,1212,0,620473,MALE,Masters,exec-managerial,basketball,unmarried,24400,-60500,28-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,NY,Springfield,7705 Best Ridge,4,1,YES,0,1,YES,73500,7350,14700,51450,Jeep,Wrangler,1999,N
371,52,740505,12-10-1997,IL,250/500,1000,1242.96,7000000,449800,FEMALE,High School,other-service,paintball,own-child,0,-37100,22-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Riverwood,5838 Pine Lane,2,2,YES,2,0,?,88920,6840,13680,68400,Accura,RSX,2010,N
343,52,629663,21-01-2002,IL,500/1000,1000,1053.02,0,602402,FEMALE,Associate,prof-specialty,bungie-jumping,not-in-family,0,0,16-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Arlington,7331 Sky Hwy,2,1,?,0,2,NO,47630,12990,4330,30310,Toyota,Corolla,2005,Y
377,53,839884,02-09-1996,IL,100/300,500,1693.63,0,452456,FEMALE,MD,craft-repair,kayaking,unmarried,0,-64000,17-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,WV,Hillsdale,5640 Embaracadero Lane,10,3,YES,1,3,NO,59040,6560,6560,45920,Saab,93,2015,N
154,37,241562,28-01-2010,IL,250/500,1000,2047.59,0,439269,FEMALE,MD,farming-fishing,dancing,other-relative,0,-67800,09-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,SC,Columbus,9610 Cherokee St,2,1,?,0,3,NO,79530,14460,7230,57840,Accura,MDX,2000,N
166,34,405533,03-10-2014,OH,100/300,1000,1083.72,0,617774,FEMALE,High School,machine-op-inspct,base-jumping,wife,65600,-68200,09-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Columbus,3550 Washington Ave,18,1,NO,0,2,YES,53680,4880,4880,43920,Honda,CRV,2005,N
298,46,667021,02-05-2007,OH,500/1000,1000,1138.42,6000000,477678,MALE,JD,prof-specialty,dancing,own-child,36900,-55000,16-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Arlington,5277 Texas Lane,18,1,NO,2,3,YES,33550,3050,6100,24400,Volkswagen,Passat,2005,N
235,42,511621,22-09-1990,IN,250/500,500,1072.62,0,444913,FEMALE,Masters,machine-op-inspct,exercise,husband,39900,-60200,18-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Springfield,3654 Cherokee Ave,7,4,?,2,1,YES,69100,6910,6910,55280,Toyota,Highlander,2006,N
172,35,476923,19-09-2004,IL,100/300,2000,1219.04,0,456602,MALE,MD,handlers-cleaners,paintball,own-child,63600,-68700,11-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,OH,Hillsdale,7380 5th Hwy,19,3,?,2,0,NO,79750,14500,14500,50750,Nissan,Pathfinder,1999,N
27,28,735822,28-08-1995,IN,100/300,2000,1371.78,0,451560,MALE,JD,farming-fishing,polo,other-relative,0,-32500,04-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,PA,Riverwood,2539 Embaracadero Ridge,10,3,?,0,2,YES,53600,5360,10720,37520,Audi,A5,2015,Y
428,54,492745,04-02-2004,IN,100/300,2000,1506.21,0,453407,MALE,Masters,transport-moving,kayaking,unmarried,0,-24400,22-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,NY,Riverwood,4693 Lincoln Hwy,16,3,NO,1,0,?,76560,12760,6380,57420,Nissan,Ultima,2009,N
99,32,130930,23-07-2014,IN,100/300,1000,1058.21,3000000,618655,MALE,JD,craft-repair,golf,unmarried,0,0,10-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Arlington,2376 Sky Ridge,17,1,YES,1,1,NO,41130,4570,4570,31990,Dodge,RAM,1999,N
107,26,261119,21-03-1997,IL,500/1000,2000,932.14,0,612550,MALE,MD,sales,cross-fit,own-child,40600,0,10-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Columbus,1273 Rock Lane,2,3,NO,2,3,NO,78650,14300,7150,57200,Audi,A3,1996,Y
272,41,280709,06-05-1991,OH,500/1000,2000,1608.34,0,466718,FEMALE,Associate,farming-fishing,reading,own-child,33300,-10600,16-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NY,Springfield,8281 Lincoln Lane,19,1,?,1,1,?,71060,6460,12920,51680,Saab,92x,2010,N
151,37,898573,07-08-1992,IN,500/1000,1000,1728.56,0,617947,FEMALE,Masters,farming-fishing,dancing,own-child,54000,0,08-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbrook,6429 4th Hwy,0,1,YES,1,1,?,38830,3530,3530,31770,Suburu,Legacy,1999,N
249,43,547802,03-09-2013,IL,250/500,1000,1518.46,0,606238,FEMALE,MD,armed-forces,cross-fit,own-child,0,0,26-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Riverwood,2201 4th Lane,16,1,?,0,0,YES,53500,5350,5350,42800,Saab,92x,2015,N
177,38,600845,05-01-2012,IL,100/300,2000,1540.19,0,463842,FEMALE,College,adm-clerical,skydiving,other-relative,0,-74500,01-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,VA,Columbus,5506 Best St,20,1,YES,2,0,NO,73700,7370,7370,58960,Chevrolet,Silverado,2001,Y
190,40,390381,27-01-2007,OH,500/1000,2000,965.21,0,610354,FEMALE,JD,exec-managerial,camping,other-relative,36900,-53700,02-02-2015,Parked Car,?,Trivial Damage,None,SC,Hillsdale,8404 Embaracadero St,10,1,?,2,1,YES,6300,630,630,5040,Nissan,Ultima,2001,N
174,36,629918,14-10-2005,IL,100/300,2000,1278.75,0,461328,FEMALE,College,tech-support,paintball,own-child,53200,-53800,06-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,NY,Riverwood,2117 Lincoln Hwy,21,3,?,2,2,NO,65400,10900,10900,43600,Dodge,RAM,2012,N
95,28,208298,03-11-1990,OH,250/500,1000,773.99,0,458727,MALE,Associate,armed-forces,board-games,other-relative,0,-70300,01-01-2015,Vehicle Theft,?,Trivial Damage,None,PA,Springfield,6359 MLK Ridge,3,1,YES,1,2,NO,3200,640,320,2240,Mercedes,E400,2014,N
371,51,513099,15-10-2005,IN,500/1000,1000,1532.47,0,452587,FEMALE,Associate,tech-support,golf,other-relative,60300,-24700,19-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NY,Arlington,9751 Sky Ridge,7,1,YES,0,3,YES,75400,7540,15080,52780,Dodge,RAM,2012,Y
2,28,184938,22-05-1999,IL,250/500,1000,1340.56,0,433184,FEMALE,JD,machine-op-inspct,golf,not-in-family,0,0,17-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,NY,Northbend,9020 Elm Ave,19,1,YES,0,2,YES,58140,6460,6460,45220,Saab,92x,2008,N
269,44,187775,21-12-2002,OH,100/300,500,1297.75,4000000,451280,FEMALE,JD,other-service,chess,own-child,0,-41400,01-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Columbus,1830 Sky St,15,3,?,0,1,?,98670,15180,15180,68310,Chevrolet,Tahoe,2010,Y
101,27,326322,10-02-2007,IL,250/500,1000,433.33,0,603269,MALE,Masters,machine-op-inspct,golf,other-relative,25900,0,02-01-2015,Parked Car,?,Minor Damage,None,SC,Hillsdale,6067 Weaver Ridge,7,1,?,0,3,NO,5900,1180,590,4130,Mercedes,E400,2009,N
94,30,146138,01-03-2002,IN,250/500,2000,1025.54,0,442632,FEMALE,High School,armed-forces,paintball,other-relative,0,-52600,20-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NC,Arlington,1840 Embaracadero Ave,19,1,?,1,3,YES,64100,6410,6410,51280,Chevrolet,Malibu,2001,N
117,28,336047,21-04-2003,OH,250/500,500,1264.77,0,447300,FEMALE,Associate,transport-moving,yachting,unmarried,47500,-32500,05-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Springfield,4058 Tree Drive,13,2,YES,0,1,?,55440,5040,10080,40320,Suburu,Forrestor,2009,Y
111,27,532330,22-09-2002,OH,250/500,500,1459.97,5000000,441783,MALE,MD,sales,yachting,other-relative,0,0,27-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,SC,Northbend,4983 MLK Ridge,2,3,NO,1,2,NO,80850,7350,14700,58800,Ford,F150,2011,Y
242,40,118137,10-02-1998,OH,100/300,500,1238.65,0,468702,FEMALE,High School,transport-moving,bungie-jumping,husband,0,-44600,27-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Springfield,9744 Texas Drive,5,1,YES,1,1,NO,7480,680,680,6120,Saab,95,1998,N
440,61,212674,01-09-1992,OH,250/500,500,1050.76,0,467942,MALE,PhD,transport-moving,movies,wife,41500,-70200,26-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,OH,Riverwood,8821 Elm St,21,1,?,2,3,NO,53640,5960,5960,41720,Nissan,Maxima,2004,Y
20,23,935596,01-05-1999,OH,500/1000,1000,1711.72,0,463678,FEMALE,JD,tech-support,base-jumping,wife,0,0,13-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Northbrook,2886 Tree Ridge,21,3,NO,2,2,NO,63250,5750,11500,46000,Chevrolet,Tahoe,2002,Y
461,57,737593,19-12-1997,IL,100/300,500,865.33,7000000,615220,FEMALE,High School,farming-fishing,golf,own-child,0,0,14-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Hillsdale,5236 Weaver Drive,7,3,NO,0,1,?,59040,9840,9840,39360,Dodge,RAM,1995,N
208,36,812025,18-06-2000,IL,250/500,500,1153.49,0,432711,MALE,Associate,craft-repair,base-jumping,other-relative,0,0,22-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,SC,Arlington,5862 Apache Ridge,16,3,YES,1,0,YES,50500,10100,5050,35350,Accura,TL,2004,N
279,43,168151,24-04-1995,OH,500/1000,2000,1281.25,0,463583,MALE,Associate,machine-op-inspct,cross-fit,other-relative,0,0,17-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,SC,Columbus,7859 4th Ridge,13,1,YES,0,1,NO,57690,6410,12820,38460,Mercedes,C300,2010,Y
244,40,594739,16-06-2006,IL,100/300,500,1342.8,0,439502,FEMALE,MD,sales,base-jumping,husband,0,0,02-02-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Springfield,6259 Weaver St,2,1,YES,1,3,YES,5940,660,660,4620,Toyota,Camry,2014,N
134,30,843227,28-09-2007,OH,250/500,2000,1443.32,0,613287,FEMALE,PhD,exec-managerial,dancing,unmarried,0,0,07-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Northbend,9980 Lincoln Ave,19,3,?,0,2,YES,47790,5310,5310,37170,Suburu,Impreza,1995,Y
122,29,283925,21-11-1991,OH,250/500,1000,1629.94,0,620104,FEMALE,Masters,priv-house-serv,skydiving,other-relative,0,-47100,02-02-2015,Parked Car,?,Trivial Damage,None,SC,Riverwood,7828 Cherokee Ave,17,1,?,0,1,NO,3850,350,350,3150,Dodge,Neon,2014,N
156,31,475588,21-09-1996,IL,100/300,2000,1134.08,0,446895,MALE,PhD,other-service,reading,husband,0,0,07-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NC,Arlington,5812 Oak St,3,1,?,2,0,?,59000,5900,5900,47200,Ford,Fusion,2013,Y
232,43,751905,16-05-2001,OH,250/500,500,1483.91,8000000,431531,MALE,College,machine-op-inspct,golf,husband,0,-33600,18-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,NY,Arlington,2318 Washington Hwy,17,3,NO,0,1,?,70600,7060,14120,49420,Volkswagen,Passat,2013,Y
244,40,226725,11-08-1999,IN,500/1000,2000,1304.67,7000000,605408,MALE,Masters,armed-forces,base-jumping,other-relative,0,-45000,10-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,SC,Hillsdale,8809 Flute St,5,3,?,1,1,?,61490,5590,11180,44720,Dodge,RAM,2001,N
84,30,942504,16-06-2003,IL,500/1000,2000,1035.79,0,457551,FEMALE,MD,protective-serv,kayaking,wife,44400,-51500,30-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Northbend,3184 Oak Ave,9,1,NO,1,0,YES,57640,5240,10480,41920,Volkswagen,Passat,2010,N
394,57,395572,30-03-1999,IL,250/500,500,1401.2,0,619892,FEMALE,High School,craft-repair,movies,own-child,51500,0,25-01-2015,Vehicle Theft,?,Trivial Damage,Police,NC,Hillsdale,6493 Lincoln Lane,9,1,NO,1,0,NO,6890,1060,1060,4770,Audi,A5,1999,N
246,45,889883,03-02-1999,IL,250/500,1000,1665.45,0,445853,MALE,JD,machine-op-inspct,hiking,wife,34400,-33100,29-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,PA,Columbus,7162 Maple Ave,9,3,?,2,2,?,53280,11840,5920,35520,Toyota,Camry,2006,Y
35,29,818167,25-08-2011,IN,500/1000,2000,653.66,0,475483,FEMALE,JD,handlers-cleaners,video-games,unmarried,52100,-46900,24-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Springfield,5455 Tree Ridge,22,3,NO,0,0,?,78300,15660,7830,54810,BMW,X5,2009,N
156,37,277767,28-06-2010,OH,100/300,500,1080.13,0,606290,MALE,Associate,protective-serv,reading,other-relative,0,-61000,04-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,PA,Springfield,5778 Pine Ridge,15,3,YES,0,3,NO,41490,4610,4610,32270,Nissan,Pathfinder,2001,N
195,36,842618,06-11-2001,IN,100/300,2000,1346.18,0,611852,FEMALE,Associate,machine-op-inspct,camping,wife,57800,-53300,25-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,SC,Hillsdale,3797 Solo Lane,14,3,YES,2,3,YES,68970,12540,6270,50160,Chevrolet,Tahoe,2007,N
369,55,577810,15-04-2013,OH,250/500,2000,1589.54,0,444734,MALE,College,handlers-cleaners,camping,husband,55400,0,27-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,VA,Arlington,9373 Pine Hwy,6,3,?,2,0,YES,85300,17060,8530,59710,Toyota,Highlander,2003,N
271,40,873114,07-12-1995,IL,100/300,1000,1251.65,0,433683,FEMALE,Associate,other-service,camping,wife,71200,0,19-02-2015,Parked Car,?,Minor Damage,None,NY,Hillsdale,1365 Francis Ave,6,1,NO,0,0,NO,3080,560,280,2240,Audi,A3,2012,N
332,47,994538,01-11-1991,IL,100/300,2000,1083.01,0,448882,MALE,MD,craft-repair,paintball,other-relative,91900,0,31-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Police,WV,Arlington,9239 Washington Ridge,22,4,YES,2,0,?,71760,11040,11040,49680,Jeep,Grand Cherokee,2010,Y
107,26,727792,19-05-2014,OH,100/300,500,974.59,0,466838,FEMALE,JD,armed-forces,skydiving,wife,62800,0,18-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Arlington,3416 Washington Drive,14,3,?,1,0,NO,59700,11940,11940,35820,Nissan,Ultima,2002,N
217,39,522506,15-03-1992,IL,500/1000,2000,1399.85,0,605490,FEMALE,Masters,other-service,skydiving,other-relative,49900,-19800,10-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Columbus,1923 2nd Hwy,16,3,NO,0,2,?,64920,10820,10820,43280,Ford,Fusion,1997,N
243,43,367595,03-02-2006,IN,500/1000,500,1307.74,0,466137,FEMALE,Associate,machine-op-inspct,board-games,own-child,0,-75700,28-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,SC,Riverwood,6451 1st Hwy,10,3,?,0,1,NO,37530,4170,4170,29190,Jeep,Wrangler,2008,N
296,42,586104,16-03-2003,IN,250/500,2000,1219.27,0,466970,MALE,Associate,tech-support,paintball,husband,53100,-63400,16-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,WV,Columbus,1267 Francis Hwy,9,3,YES,1,2,NO,64080,7120,7120,49840,Saab,93,2012,N
264,41,424862,16-10-2002,OH,100/300,500,1411.3,0,474801,MALE,PhD,prof-specialty,cross-fit,unmarried,55600,0,08-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,PA,Northbrook,4158 Washington Lane,4,1,NO,1,0,?,60390,10980,5490,43920,BMW,M5,2004,Y
108,33,512813,27-01-1990,IL,100/300,2000,694.45,0,450703,FEMALE,JD,armed-forces,exercise,not-in-family,0,0,20-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,WV,Northbend,3796 Cherokee Drive,6,3,?,0,1,YES,64350,5850,11700,46800,Nissan,Pathfinder,2011,Y
32,38,356768,11-03-2010,IL,100/300,500,1006.77,6000000,478172,FEMALE,College,other-service,sleeping,own-child,0,0,06-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,SC,Northbend,7434 Oak Hwy,15,3,YES,2,1,YES,70900,14180,7090,49630,BMW,X6,1997,N
259,39,330506,19-09-1995,OH,250/500,1000,1422.36,0,604668,FEMALE,JD,craft-repair,movies,unmarried,0,-83900,24-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,PA,Columbus,5178 Weaver Hwy,12,3,NO,1,3,?,46560,7760,7760,31040,Nissan,Ultima,2012,N
186,33,779075,27-02-2010,IN,100/300,1000,1348.32,0,469429,FEMALE,Associate,craft-repair,cross-fit,wife,37600,-37600,14-01-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Arlington,8477 Francis Hwy,3,1,NO,2,1,?,4730,860,860,3010,Chevrolet,Malibu,2013,Y
201,40,799501,28-12-1991,OH,250/500,2000,1315.56,0,471806,FEMALE,PhD,transport-moving,video-games,not-in-family,0,0,18-02-2015,Vehicle Theft,?,Minor Damage,Police,SC,Northbrook,7693 Britain Lane,14,1,YES,0,0,YES,6820,1240,1240,4340,Jeep,Grand Cherokee,2003,N
436,58,987905,30-04-2002,OH,250/500,2000,1407.01,5000000,475705,MALE,PhD,tech-support,sleeping,other-relative,47400,-27600,10-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,WV,Riverwood,3658 Rock Drive,10,1,?,0,2,?,59900,11980,5990,41930,BMW,X5,1997,Y
189,36,967756,28-04-2007,OH,250/500,2000,1388.58,0,459122,FEMALE,MD,priv-house-serv,basketball,own-child,0,-49400,13-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,WV,Riverwood,2617 Andromedia Drive,17,3,?,1,3,YES,79560,13260,13260,53040,Ford,Escape,2009,N
105,33,830414,08-07-1996,IL,500/1000,500,1310.76,0,476737,FEMALE,High School,adm-clerical,kayaking,not-in-family,0,-40900,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,VA,Northbrook,9279 Oak Hwy,8,1,YES,0,1,?,70290,12780,6390,51120,Accura,MDX,2008,N
163,31,127313,01-04-2002,IN,100/300,1000,1004.63,6000000,460359,MALE,JD,priv-house-serv,hiking,not-in-family,26900,0,07-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Hillsdale,3439 Andromedia Hwy,14,3,YES,2,0,NO,63910,5810,11620,46480,Suburu,Forrestor,1999,N
219,40,786957,29-10-2006,OH,100/300,500,1134.91,0,452735,FEMALE,Associate,transport-moving,golf,not-in-family,68700,0,29-01-2015,Vehicle Theft,?,Trivial Damage,None,SC,Riverwood,5901 Elm Drive,6,1,?,1,0,NO,6400,640,640,5120,Toyota,Camry,1997,N
88,25,332892,25-10-2007,IN,250/500,1000,1194,0,613583,FEMALE,JD,handlers-cleaners,movies,husband,0,0,13-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Northbrook,3982 Washington Hwy,6,1,YES,1,2,YES,66780,7420,7420,51940,Ford,Escape,2013,Y
40,39,448642,28-03-2001,IN,500/1000,1000,1248.25,4000000,605692,FEMALE,College,sales,hiking,own-child,0,-33300,01-02-2015,Parked Car,?,Minor Damage,Police,VA,Northbrook,3376 5th Drive,8,1,NO,0,2,?,8760,1460,1460,5840,BMW,3 Series,2013,N
284,42,526039,04-05-1995,OH,100/300,500,1338.54,-1000000,438178,MALE,Associate,machine-op-inspct,kayaking,wife,0,0,29-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,NC,Arlington,3936 Tree Drive,13,1,YES,0,1,?,94160,8560,17120,68480,Chevrolet,Malibu,1996,N
59,40,444422,28-09-2011,IL,250/500,2000,782.23,0,449221,MALE,College,protective-serv,golf,other-relative,64200,-32300,06-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,OH,Springfield,6605 Tree Ave,0,3,?,0,2,NO,51570,5730,11460,34380,BMW,X5,2010,N
39,31,689500,28-01-2003,IL,250/500,2000,1366.9,0,459322,FEMALE,High School,handlers-cleaners,polo,husband,0,-15700,28-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NC,Northbend,3102 Apache St,14,3,?,1,0,NO,52700,10540,10540,31620,BMW,X6,2014,Y
147,34,806081,01-02-2011,IL,500/1000,1000,1275.81,0,472657,MALE,High School,sales,dancing,wife,0,-48300,21-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,NY,Columbus,7756 Pine Hwy,13,1,YES,2,2,?,101010,15540,15540,69930,Mercedes,ML350,1998,N
156,37,384618,09-02-1993,IN,250/500,500,1090.65,0,608331,MALE,High School,exec-managerial,golf,unmarried,0,-51800,16-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NC,Columbus,7142 5th Lane,16,4,?,1,2,YES,53400,5340,5340,42720,Honda,Civic,2010,Y
123,31,756459,05-08-2005,IN,250/500,500,1326,0,438546,FEMALE,Associate,prof-specialty,basketball,wife,0,-54600,17-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,SC,Northbrook,2914 Oak Drive,13,1,?,1,2,NO,72120,12020,12020,48080,Mercedes,ML350,2009,N
231,43,655787,17-06-2006,IL,250/500,2000,972.47,0,441981,MALE,College,protective-serv,reading,wife,0,-58100,01-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,PA,Hillsdale,6522 Apache Drive,15,3,YES,2,2,YES,77100,7710,15420,53970,Audi,A3,2010,N
247,39,419954,07-12-1993,IL,100/300,500,806.31,0,602177,FEMALE,College,handlers-cleaners,dancing,wife,0,0,25-02-2015,Parked Car,?,Trivial Damage,Police,SC,Northbend,5279 Pine Ridge,3,1,?,2,3,?,3300,600,0,2700,Dodge,RAM,2003,N
194,35,275092,14-03-2012,IL,500/1000,500,1416.24,0,441659,FEMALE,MD,adm-clerical,golf,not-in-family,0,0,26-02-2015,Parked Car,?,Minor Damage,None,WV,Northbend,8078 Britain Hwy,7,1,YES,1,0,?,5940,1080,540,4320,Nissan,Pathfinder,2003,N
119,27,515698,05-08-1997,IN,250/500,2000,1097.64,0,614812,MALE,High School,transport-moving,video-games,other-relative,27100,0,06-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,PA,Northbend,1133 Apache St,16,2,NO,1,0,?,63720,7080,21240,35400,Accura,TL,2006,N
259,43,132871,05-07-2009,IL,100/300,500,947.75,0,458470,FEMALE,Masters,farming-fishing,base-jumping,not-in-family,0,-39300,03-02-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Riverwood,2873 Flute Ave,15,1,NO,1,3,NO,7680,1280,640,5760,Audi,A5,2008,N
107,31,714929,25-11-1994,IL,100/300,2000,1018.73,5000000,469646,MALE,Associate,handlers-cleaners,yachting,own-child,20000,-82700,27-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,NY,Riverwood,8509 Apache St,21,3,YES,1,2,?,93730,14420,21630,57680,Honda,CRV,2001,N
48,44,297816,03-02-1997,IL,100/300,2000,1400.74,0,611118,MALE,College,sales,base-jumping,other-relative,34000,-55600,21-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NC,Arlington,8245 4th Hwy,23,3,YES,0,2,?,87300,17460,17460,52380,Jeep,Wrangler,2013,N
267,40,426708,09-10-2009,IL,250/500,500,1155.53,5000000,465158,MALE,JD,transport-moving,camping,wife,0,-35200,24-01-2015,Parked Car,?,Trivial Damage,Police,NY,Columbus,3094 Best Lane,5,1,?,0,2,NO,5670,1260,630,3780,Ford,F150,1997,N
286,47,615047,20-11-2002,IN,250/500,500,1386.93,0,457130,MALE,High School,priv-house-serv,bungie-jumping,husband,54100,-77600,17-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Springfield,8188 Tree Ave,15,3,YES,0,0,NO,65800,13160,6580,46060,Accura,TL,2001,N
175,34,771236,29-05-1995,OH,100/300,500,915.29,0,607893,FEMALE,JD,handlers-cleaners,base-jumping,wife,82400,-57100,23-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Arlington,5224 5th Lane,14,1,?,2,1,YES,36720,4080,4080,28560,Honda,Accord,2009,Y
111,29,235869,22-01-2011,IL,250/500,500,1239.55,2000000,464736,FEMALE,PhD,farming-fishing,kayaking,own-child,0,0,09-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,SC,Riverwood,2230 1st St,1,1,?,1,1,YES,52800,4800,9600,38400,Mercedes,ML350,1996,Y
151,37,931625,18-10-2012,IN,250/500,500,1366.42,0,476198,FEMALE,Associate,protective-serv,cross-fit,unmarried,44000,0,15-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Arlington,6719 Flute St,14,3,NO,1,2,YES,59100,5910,5910,47280,Nissan,Maxima,1998,Y
156,37,371635,13-10-1991,OH,500/1000,1000,1086.48,6000000,444903,MALE,Associate,machine-op-inspct,hiking,unmarried,0,-53800,16-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,NY,Northbrook,8064 4th Ave,17,3,?,1,1,YES,77440,14080,7040,56320,Suburu,Legacy,1999,N
165,36,427199,01-10-2010,IL,250/500,2000,1247.87,0,464336,MALE,Masters,armed-forces,golf,husband,0,-39700,14-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,NY,Northbend,2469 Francis Lane,20,3,NO,2,2,NO,45700,4570,4570,36560,BMW,3 Series,2008,N
253,41,261315,10-04-2013,OH,100/300,2000,1312.75,0,471453,FEMALE,PhD,sales,dancing,other-relative,81300,0,01-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Springfield,4671 5th Ridge,10,3,NO,2,2,YES,80740,7340,14680,58720,Toyota,Camry,2014,Y
10,26,582973,11-06-2008,IN,100/300,2000,765.64,0,466191,MALE,MD,sales,base-jumping,not-in-family,0,-22200,16-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,SC,Northbrook,6985 Maple Lane,3,3,?,0,3,?,31350,2850,5700,22800,Honda,Accord,2001,Y
158,33,278091,04-12-2013,OH,100/300,2000,1327.41,0,440930,FEMALE,Associate,handlers-cleaners,skydiving,other-relative,0,-38600,04-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,SC,Hillsdale,7791 Britain Ridge,0,1,?,0,0,?,35000,3500,7000,24500,Suburu,Legacy,2012,N
436,59,153154,21-08-2010,OH,500/1000,1000,1338.55,0,430380,MALE,PhD,protective-serv,board-games,own-child,39000,0,12-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,WV,Hillsdale,6355 4th Hwy,10,3,?,2,2,NO,68000,13600,6800,47600,Toyota,Corolla,2014,N
91,30,515217,18-06-2010,IL,250/500,2000,1316.63,8000000,613178,FEMALE,Masters,machine-op-inspct,golf,unmarried,43900,0,08-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,SC,Northbend,3495 Britain Drive,13,3,YES,2,0,?,84500,13000,13000,58500,BMW,X6,2009,N
256,42,860497,10-04-1992,IL,500/1000,1000,1286.44,0,460564,FEMALE,MD,transport-moving,bungie-jumping,wife,0,-39500,22-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,WV,Northbend,2980 Sky Ridge,14,1,NO,2,1,?,75500,7550,15100,52850,Nissan,Ultima,1998,N
274,46,351741,03-02-1997,OH,500/1000,1000,1372.18,0,439929,MALE,High School,exec-managerial,bungie-jumping,not-in-family,0,0,13-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Riverwood,5914 Oak Ave,22,3,NO,1,3,?,90600,15100,15100,60400,BMW,X5,2009,N
275,45,403737,06-12-1991,IN,500/1000,2000,1447.77,0,605756,FEMALE,Associate,adm-clerical,camping,wife,39400,-63900,18-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,VA,Northbend,3835 5th Ave,8,3,YES,1,1,?,64320,5360,10720,48240,Accura,MDX,1998,N
1,33,162004,19-09-1995,IL,250/500,500,903.32,0,451184,FEMALE,High School,transport-moving,yachting,not-in-family,0,0,19-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Northbend,5925 Tree Hwy,1,3,?,1,0,?,31700,6340,3170,22190,Toyota,Highlander,2006,N
85,30,740384,29-10-1993,IN,500/1000,1000,1454.42,0,459588,MALE,Associate,protective-serv,reading,other-relative,51600,-73900,31-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NY,Springfield,6250 1st Ridge,19,1,YES,0,1,YES,74280,12380,12380,49520,Suburu,Forrestor,2006,Y
233,37,876714,03-11-1991,IL,100/300,2000,1603.42,0,616637,FEMALE,High School,sales,video-games,wife,61600,-30200,06-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,1346 5th Lane,17,1,YES,0,2,YES,80520,13420,6710,60390,Toyota,Corolla,2005,N
142,30,951543,09-07-2002,IN,250/500,2000,1616.58,0,447979,MALE,JD,adm-clerical,polo,husband,58500,-46800,04-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Hillsdale,1128 Maple Lane,13,3,YES,0,3,YES,63600,6360,12720,44520,Dodge,RAM,2010,N
266,44,576723,07-12-1999,IL,250/500,500,1611.83,0,460176,MALE,High School,handlers-cleaners,movies,husband,0,0,02-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,WV,Riverwood,6309 Cherokee Ave,4,1,?,1,2,YES,32800,3280,3280,26240,BMW,3 Series,2012,N
350,50,391003,01-07-2005,OH,500/1000,500,889.13,0,459429,FEMALE,Masters,priv-house-serv,board-games,other-relative,0,0,26-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,PA,Northbend,4618 Flute Ave,14,3,NO,0,2,NO,44190,9820,4910,29460,Audi,A3,2015,N
97,26,225865,04-11-1991,IL,250/500,1000,1252.08,0,465456,MALE,College,exec-managerial,sleeping,not-in-family,0,0,08-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,VA,Springfield,6191 Oak Lane,4,2,YES,2,2,NO,50400,10080,5040,35280,Honda,CRV,2000,Y
399,55,984948,14-04-1993,IL,500/1000,2000,995.56,0,464665,MALE,JD,tech-support,sleeping,not-in-family,0,-65400,07-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,NY,Northbend,1316 Britain Ridge,23,3,YES,1,1,YES,88400,17680,8840,61880,Nissan,Pathfinder,2010,N
305,49,890328,23-08-2009,IL,100/300,2000,1347.92,0,430853,FEMALE,High School,farming-fishing,bungie-jumping,own-child,0,-42100,17-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,NY,Hillsdale,5924 Maple Drive,21,3,YES,0,2,?,66550,12100,6050,48400,Volkswagen,Jetta,2003,N
276,47,803294,18-06-1993,IN,100/300,1000,1724.09,0,615712,MALE,PhD,craft-repair,yachting,own-child,0,0,12-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Arlington,8917 Tree Ridge,23,3,YES,0,0,?,65780,5980,11960,47840,Chevrolet,Tahoe,2014,N
257,40,414913,17-07-2012,IN,250/500,500,1379.93,0,608228,MALE,MD,armed-forces,base-jumping,husband,0,0,01-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,NY,Columbus,3966 Francis Ridge,6,3,?,0,2,YES,51810,9420,4710,37680,Audi,A3,2002,Y
78,31,414519,25-01-1999,IN,250/500,1000,1554.64,4000000,457535,MALE,PhD,protective-serv,board-games,own-child,0,-27900,03-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,WV,Springfield,1507 Solo Ave,21,1,NO,1,0,NO,55660,5060,10120,40480,Honda,CRV,2009,Y
129,28,818413,23-02-1990,OH,500/1000,1000,1377.94,0,442540,MALE,Masters,machine-op-inspct,base-jumping,not-in-family,0,0,21-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,NY,Springfield,4272 Oak Ridge,23,1,?,2,3,?,44640,9920,4960,29760,Toyota,Camry,2005,N
283,46,487356,30-08-2000,IL,500/1000,2000,1313.33,0,455332,FEMALE,PhD,transport-moving,reading,husband,53500,-73600,09-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Riverwood,4434 Lincoln Ave,3,3,?,1,3,NO,77660,7060,14120,56480,Nissan,Maxima,2004,Y
85,25,159768,03-09-2008,IN,250/500,500,1259.02,0,439534,FEMALE,JD,tech-support,base-jumping,unmarried,67000,-53600,16-02-2015,Parked Car,?,Trivial Damage,None,SC,Northbend,7529 Solo Ridge,8,1,NO,2,2,?,5640,940,940,3760,Nissan,Ultima,2005,N
101,26,865839,02-08-1991,IL,500/1000,1000,1371.88,0,462420,FEMALE,MD,prof-specialty,reading,husband,0,0,04-02-2015,Parked Car,?,Trivial Damage,None,VA,Arlington,8096 Apache Hwy,4,1,?,2,2,?,3190,580,580,2030,Suburu,Legacy,1995,N
96,30,406567,25-09-2001,OH,100/300,500,1399.27,6000000,448913,MALE,College,prof-specialty,hiking,wife,38900,-48700,24-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,NC,Arlington,9417 Tree Hwy,22,1,?,0,0,YES,53440,0,6680,46760,Ford,Escape,2004,N
121,31,623032,11-03-2007,IL,500/1000,1000,1061.98,6000000,440837,FEMALE,JD,armed-forces,camping,unmarried,0,0,01-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,WV,Hillsdale,3809 Texas Lane,16,1,NO,0,1,YES,65250,7250,7250,50750,BMW,3 Series,2002,N
176,39,935442,20-11-2010,OH,250/500,500,1365.46,4000000,466634,MALE,College,armed-forces,sleeping,unmarried,0,-56600,05-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,SC,Columbus,1540 Apache Lane,14,3,NO,2,1,NO,44280,4920,4920,34440,Chevrolet,Silverado,2008,N
159,37,106873,28-08-1998,IL,500/1000,1000,894.4,0,446435,MALE,Associate,tech-support,camping,wife,0,-53700,07-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,NY,Springfield,2337 Lincoln Hwy,13,1,YES,2,0,NO,70290,7810,7810,54670,Dodge,RAM,1999,N
120,30,563878,16-07-2002,IN,250/500,500,956.69,0,438237,FEMALE,Associate,priv-house-serv,movies,husband,39600,-64300,06-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,NY,Hillsdale,6770 1st St,20,1,?,1,1,YES,87100,8710,8710,69680,Saab,92x,2000,N
212,35,620855,29-04-1990,IN,500/1000,2000,1123.89,0,468313,MALE,MD,priv-house-serv,video-games,unmarried,35400,-49200,21-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,4119 Texas St,0,3,?,1,3,?,50380,4580,4580,41220,Suburu,Forrestor,1996,N
290,45,583169,01-02-1998,IL,100/300,500,1085.03,0,476303,FEMALE,JD,sales,cross-fit,wife,0,-61000,01-03-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,VA,Arlington,4347 2nd Ridge,23,3,YES,2,2,NO,64800,12960,6480,45360,Audi,A3,2014,Y
299,42,337677,20-07-2008,OH,100/300,2000,1437.33,0,450339,FEMALE,Associate,craft-repair,movies,wife,25000,0,24-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,SC,Columbus,1091 1st Drive,13,1,YES,1,3,NO,70400,12800,12800,44800,BMW,3 Series,2000,N
66,26,445973,13-11-1998,IL,250/500,1000,988.29,0,476502,MALE,College,armed-forces,skydiving,own-child,0,0,02-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,NY,Northbend,8203 Lincoln Ave,8,1,?,2,2,YES,57860,0,10520,47340,Suburu,Impreza,2008,Y
334,47,156694,24-05-2001,IL,500/1000,500,1238.89,0,600561,MALE,Masters,protective-serv,sleeping,other-relative,0,0,31-01-2015,Vehicle Theft,?,Minor Damage,None,WV,Northbend,9154 MLK Hwy,3,1,?,0,3,NO,6240,960,960,4320,Ford,Fusion,2011,N
216,38,421940,03-06-2014,IN,100/300,1000,1384.64,5000000,600754,FEMALE,Associate,tech-support,board-games,unmarried,0,0,09-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,VA,Columbus,5780 4th Ave,16,1,?,2,3,NO,66600,16650,11100,38850,Jeep,Grand Cherokee,2012,Y
86,28,613226,22-08-1991,IN,100/300,2000,1595.07,0,439304,MALE,PhD,transport-moving,hiking,unmarried,75800,0,23-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,VA,Hillsdale,6945 Texas Hwy,19,1,YES,0,2,YES,70920,11820,11820,47280,Jeep,Wrangler,2002,N
429,56,804410,12-12-1998,OH,250/500,1000,1127.89,6000000,460722,MALE,Associate,machine-op-inspct,skydiving,own-child,67400,-43800,28-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Springfield,5639 1st Ridge,0,1,YES,2,0,YES,39480,6580,6580,26320,Suburu,Forrestor,2002,N
257,43,553565,18-02-1999,IN,500/1000,2000,929.7,6000000,618632,FEMALE,PhD,handlers-cleaners,base-jumping,husband,46400,-74300,14-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,VA,Arlington,3834 Pine St,12,1,?,2,2,YES,63240,10540,5270,47430,Mercedes,E400,2005,N
15,34,399524,30-10-1997,IL,100/300,1000,1829.63,0,452204,MALE,JD,tech-support,cross-fit,not-in-family,56700,0,03-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,SC,Arlington,1358 Maple St,21,3,YES,1,0,?,67650,12300,6150,49200,Audi,A5,2009,N
230,39,331595,29-11-1999,IL,100/300,1000,904.7,7000000,454530,FEMALE,MD,craft-repair,bungie-jumping,unmarried,68600,-22300,17-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Riverwood,7460 Apache Lane,0,1,?,1,3,NO,74200,14840,14840,44520,Accura,TL,2002,Y
250,43,380067,07-07-2013,OH,500/1000,1000,1243.84,0,474848,FEMALE,JD,tech-support,polo,own-child,47900,-73400,30-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,VA,Columbus,5771 Sky Ave,22,1,YES,1,3,NO,64900,12980,12980,38940,Volkswagen,Jetta,2011,N
270,44,701521,05-07-2003,IL,500/1000,2000,1030.95,0,435985,FEMALE,Associate,machine-op-inspct,paintball,other-relative,47200,0,03-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,NC,Northbend,2865 Maple Lane,20,3,?,1,0,NO,35900,7180,3590,25130,Audi,A3,2007,Y
65,26,360770,21-09-2005,IN,100/300,500,1285.03,3000000,457942,FEMALE,High School,craft-repair,camping,unmarried,0,-41500,03-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,VA,Riverwood,8940 Elm Ave,0,1,NO,1,3,?,52200,5220,10440,36540,Honda,CRV,2011,N
475,57,958785,18-02-1995,OH,100/300,500,1216.56,0,436522,MALE,Masters,adm-clerical,skydiving,own-child,67400,-83200,31-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,SC,Hillsdale,1215 Pine Hwy,20,1,?,0,2,NO,78000,6500,13000,58500,Suburu,Forrestor,2000,N
77,27,797934,07-04-1999,IN,500/1000,2000,966.26,0,471704,FEMALE,High School,adm-clerical,base-jumping,own-child,56400,-32800,06-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Springfield,6874 Maple Ridge,1,3,YES,0,0,?,67200,6720,6720,53760,Volkswagen,Passat,1995,N
256,43,883980,13-12-2014,OH,100/300,500,1203.17,0,455810,FEMALE,MD,prof-specialty,golf,unmarried,56700,-65600,06-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,WV,Hillsdale,8834 Elm Drive,11,1,NO,0,0,?,63250,11500,5750,46000,Nissan,Ultima,1997,N
229,37,340614,01-06-1997,IL,250/500,2000,1212.12,0,446544,FEMALE,MD,craft-repair,paintball,not-in-family,65600,0,29-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,WV,Columbus,8542 Lincoln Ridge,14,3,YES,1,1,YES,68760,11460,5730,51570,Ford,Fusion,1995,N
110,28,435784,13-07-2013,OH,250/500,1000,1573.93,0,461919,MALE,College,other-service,movies,other-relative,30400,0,07-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,NY,Springfield,9397 Francis St,20,3,YES,0,2,?,65040,10840,10840,43360,Suburu,Impreza,2010,N
177,33,563837,30-12-2002,IL,100/300,1000,1609.67,0,470128,MALE,College,adm-clerical,yachting,wife,0,-13200,20-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,SC,Springfield,4907 Andromedia Drive,22,1,?,1,3,?,82800,20700,13800,48300,Jeep,Grand Cherokee,2004,Y
292,44,200827,28-02-1997,OH,500/1000,500,1097.57,0,462836,MALE,PhD,priv-house-serv,basketball,unmarried,0,0,28-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,SC,Columbus,4429 Washington St,12,1,NO,1,0,YES,61700,6170,6170,49360,Saab,93,2005,N
451,61,533941,18-06-1998,IN,250/500,2000,1618.65,2000000,475407,FEMALE,Associate,transport-moving,polo,unmarried,0,-42600,04-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,OH,Columbus,2651 MLK Lane,3,3,YES,2,1,YES,78100,15620,7810,54670,Chevrolet,Tahoe,1997,Y
61,24,265026,08-02-1996,IN,100/300,500,922.67,0,473611,FEMALE,College,priv-house-serv,paintball,other-relative,47400,0,12-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Northbend,2942 1st Lane,15,3,YES,2,1,?,65520,9360,9360,46800,Toyota,Highlander,2011,Y
150,30,354481,17-11-2004,IN,100/300,1000,1342.02,0,608425,MALE,MD,prof-specialty,polo,own-child,0,0,28-02-2015,Parked Car,?,Trivial Damage,None,VA,Arlington,6317 Best St,8,1,YES,0,2,NO,4500,450,450,3600,Saab,93,1999,N
283,41,566720,25-10-2012,OH,100/300,500,1195.01,0,476227,FEMALE,Associate,sales,reading,own-child,60700,-54300,08-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,WV,Northbend,1555 Washington Lane,13,3,NO,0,2,?,42700,0,6100,36600,Mercedes,ML350,2011,Y
291,46,832746,13-04-2006,OH,500/1000,1000,994.74,0,452701,FEMALE,High School,adm-clerical,polo,own-child,0,-55300,25-01-2015,Parked Car,?,Minor Damage,Police,SC,Hillsdale,1919 4th Lane,8,1,NO,2,2,YES,5580,620,620,4340,Volkswagen,Passat,2005,Y
162,31,386690,21-02-2006,IN,100/300,1000,1050.24,0,456789,FEMALE,Masters,adm-clerical,chess,wife,30700,0,26-02-2015,Parked Car,?,Minor Damage,None,NC,Arlington,5480 3rd Ridge,7,1,?,0,0,NO,3600,360,720,2520,BMW,X5,2013,Y
154,36,979285,17-12-2003,IL,250/500,2000,1313.51,7000000,600904,FEMALE,Masters,exec-managerial,dancing,own-child,68500,0,03-02-2015,Vehicle Theft,?,Trivial Damage,None,SC,Northbrook,8864 Tree Ridge,9,1,?,2,0,?,2800,280,280,2240,Volkswagen,Passat,2015,N
289,47,594722,31-07-1999,OH,500/1000,2000,1102.29,0,450889,FEMALE,Associate,adm-clerical,hiking,own-child,73000,-37900,31-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,VA,Northbend,2777 Solo Drive,15,1,NO,1,0,YES,54000,6000,6000,42000,Toyota,Highlander,1996,N
10,19,216738,05-08-2014,IN,250/500,1000,1185.78,0,478837,FEMALE,JD,craft-repair,yachting,wife,0,-60700,01-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,NC,Northbend,9929 Rock Drive,5,1,?,0,2,?,48950,4450,8900,35600,Accura,TL,2011,Y
309,47,369048,05-06-2011,IL,500/1000,500,1527.95,0,611322,MALE,PhD,exec-managerial,hiking,other-relative,69400,0,21-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,WV,Northbend,4143 Maple Ridge,15,4,YES,0,1,?,77800,15560,15560,46680,Dodge,RAM,2002,N
396,57,514424,11-10-1992,IN,100/300,1000,1366.39,0,438180,MALE,High School,protective-serv,exercise,other-relative,0,-22400,30-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,NC,Columbus,7121 Rock St,22,3,YES,2,1,NO,52560,11680,5840,35040,Saab,93,1995,N
273,41,954191,17-02-2010,OH,500/1000,1000,1403.9,0,449793,FEMALE,PhD,farming-fishing,dancing,own-child,0,0,31-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,VA,Riverwood,9067 Texas Ave,16,2,?,1,2,YES,44110,4010,8020,32080,Honda,Accord,2015,N
129,30,150181,06-05-2007,IL,500/1000,2000,927.23,0,450730,FEMALE,PhD,sales,video-games,husband,51500,0,13-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,SC,Hillsdale,9245 Weaver Ridge,7,1,NO,1,3,?,74360,13520,6760,54080,Suburu,Forrestor,2009,N
140,31,388671,01-05-1997,OH,250/500,2000,1554.86,6000000,608758,FEMALE,JD,armed-forces,base-jumping,wife,59000,0,16-02-2015,Parked Car,?,Minor Damage,None,WV,Arlington,4585 Francis Ave,2,1,YES,1,2,?,6120,680,680,4760,Honda,Civic,2002,Y
419,53,457244,28-01-1998,IL,500/1000,2000,736.07,6000000,445339,MALE,College,transport-moving,chess,unmarried,45700,0,04-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,SC,Northbend,6738 Francis Hwy,17,4,?,0,0,YES,62280,5190,10380,46710,Suburu,Forrestor,2012,N
315,44,206667,05-05-1993,IL,250/500,1000,974.16,6000000,438328,FEMALE,Masters,sales,reading,other-relative,0,-56800,07-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,NY,Springfield,7576 Pine Ridge,12,3,?,1,0,YES,26730,4860,4860,17010,Volkswagen,Jetta,2006,N
72,29,745200,06-08-1994,OH,500/1000,500,973.8,0,479913,FEMALE,Associate,craft-repair,exercise,own-child,0,-85900,16-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,WV,Arlington,9105 Tree Lane,9,1,?,1,0,NO,66200,6620,6620,52960,Dodge,Neon,2013,N
32,26,412703,14-11-2014,OH,100/300,2000,1260.32,6000000,460760,MALE,JD,other-service,polo,not-in-family,0,-79800,28-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,VA,Northbrook,2299 Britain Drive,16,1,?,1,2,?,45500,9100,4550,31850,Toyota,Corolla,2009,N
230,41,736771,14-12-1991,IN,100/300,1000,1464.03,0,444797,MALE,JD,transport-moving,sleeping,own-child,0,0,08-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Springfield,1914 Francis St,19,3,?,2,0,?,53040,4420,4420,44200,Audi,A3,2006,N
157,32,347984,21-10-2009,OH,100/300,2000,617.11,0,436711,MALE,College,other-service,reading,other-relative,0,-54100,02-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,VA,Columbus,6658 Weaver St,14,3,?,1,2,NO,50800,10160,5080,35560,Mercedes,E400,2013,Y
265,41,626074,29-09-1997,IN,250/500,2000,1724.46,6000000,432491,FEMALE,Associate,craft-repair,sleeping,own-child,81800,0,13-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Northbend,1985 5th Ave,18,3,?,1,3,?,44200,4420,4420,35360,Audi,A5,2014,N
47,34,218109,31-12-2003,IL,500/1000,500,1161.31,0,617527,FEMALE,PhD,exec-managerial,base-jumping,other-relative,64800,-24300,07-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,SC,Springfield,1707 Sky Ave,23,1,YES,1,3,?,62920,11440,5720,45760,Chevrolet,Malibu,2012,N
113,29,999435,01-01-2008,OH,250/500,2000,1091.73,0,601213,MALE,PhD,exec-managerial,golf,not-in-family,36100,-42300,05-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Springfield,6456 Andromedia Drive,15,3,?,0,2,YES,49950,5550,5550,38850,Nissan,Ultima,2004,Y
289,46,858060,31-05-2004,IL,250/500,2000,1209.07,0,604138,MALE,JD,armed-forces,chess,unmarried,0,0,28-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NC,Arlington,5649 Texas Ave,18,1,YES,0,1,YES,56430,6270,6270,43890,BMW,3 Series,1995,Y
254,41,500384,18-12-2013,IL,250/500,2000,1241.04,0,431361,FEMALE,Masters,protective-serv,board-games,own-child,0,0,04-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Riverwood,1220 MLK Ave,16,1,NO,2,2,YES,100210,18220,18220,63770,Audi,A5,2014,N
115,30,903785,24-08-2000,OH,500/1000,500,1757.21,0,477695,MALE,High School,prof-specialty,base-jumping,wife,46400,0,02-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Northbend,1589 Pine St,12,3,NO,1,0,YES,49140,5460,5460,38220,Ford,F150,2007,N
236,38,873859,14-10-1992,OH,250/500,1000,802.24,0,612597,FEMALE,College,other-service,paintball,not-in-family,0,-62500,23-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,WV,Northbrook,8906 Elm Lane,16,1,NO,0,1,?,66840,16710,5570,44560,Mercedes,E400,2014,N
7,21,204294,16-11-1991,IN,500/1000,1000,1342.72,0,445638,MALE,Associate,machine-op-inspct,camping,wife,0,-45300,10-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NY,Hillsdale,2654 Elm Drive,21,1,?,1,2,?,62460,6940,6940,48580,Honda,Accord,2003,N
208,36,467106,08-10-1995,OH,100/300,2000,1209.41,5000000,476185,MALE,JD,machine-op-inspct,base-jumping,wife,0,0,16-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,NY,Columbus,6681 Texas Ridge,15,3,YES,0,1,?,62810,11420,11420,39970,Nissan,Ultima,1999,N
126,33,357713,28-10-2007,OH,500/1000,1000,1141.71,2000000,435995,FEMALE,JD,priv-house-serv,sleeping,own-child,36700,-73400,04-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,WV,Northbrook,7782 Rock St,21,1,YES,1,2,?,54160,6770,6770,40620,Suburu,Legacy,2009,N
48,35,890026,16-05-2008,IL,100/300,500,1090.03,0,430232,FEMALE,JD,exec-managerial,golf,unmarried,0,-51000,30-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,WV,Arlington,9286 Oak Ave,1,1,YES,0,2,NO,48400,9680,4840,33880,Saab,92x,2005,N
297,48,751612,22-06-2009,IN,250/500,1000,1464.73,3000000,443861,MALE,PhD,exec-managerial,golf,other-relative,54900,-36700,25-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,NY,Arlington,8758 5th St,17,3,?,0,0,NO,51480,5720,5720,40040,Toyota,Highlander,2013,N
160,36,876680,10-05-2012,OH,100/300,1000,1118.58,0,460801,FEMALE,High School,prof-specialty,board-games,husband,0,-36600,22-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,NY,Columbus,7281 Maple Hwy,5,3,NO,2,1,NO,51700,5170,10340,36190,Saab,95,2003,N
406,58,756981,02-10-2003,OH,250/500,2000,1117.04,0,605121,MALE,MD,exec-managerial,video-games,own-child,0,-42700,01-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,WV,Northbend,7571 Elm Ridge,15,3,?,1,2,?,65520,10920,5460,49140,Volkswagen,Jetta,2009,N
157,31,121439,02-08-1990,IN,500/1000,500,1257.83,7000000,458622,MALE,High School,farming-fishing,reading,own-child,40700,-41600,14-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,OH,Arlington,6738 Washington Hwy,2,4,NO,2,2,NO,47700,4770,9540,33390,Accura,TL,2011,Y
146,31,411289,16-09-1997,OH,250/500,2000,1082.72,0,478661,FEMALE,PhD,machine-op-inspct,video-games,not-in-family,61400,-57500,15-01-2015,Vehicle Theft,?,Minor Damage,None,SC,Northbend,4188 Britain Ave,3,1,YES,2,0,NO,5220,580,580,4060,Accura,MDX,2015,N
409,57,538466,29-07-1995,IN,100/300,1000,1191.8,6000000,435299,MALE,High School,protective-serv,exercise,unmarried,55600,0,06-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NY,Riverwood,6934 Lincoln Ave,19,1,NO,1,0,?,73320,6110,12220,54990,Ford,Fusion,2012,N
252,46,932097,06-09-2005,IN,100/300,1000,1242.02,0,601961,MALE,Masters,adm-clerical,dancing,wife,0,-28800,08-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,VA,Hillsdale,6390 Apache St,17,1,YES,0,2,YES,74900,14980,7490,52430,Jeep,Grand Cherokee,2003,N
6,27,463727,05-08-1992,OH,250/500,500,1075.71,0,604328,FEMALE,High School,prof-specialty,dancing,unmarried,0,-47400,17-02-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Columbus,7615 Weaver Drive,7,1,?,0,1,YES,3190,580,290,2320,Saab,95,2015,N
103,33,552618,22-01-1993,IN,100/300,1000,969.88,6000000,614385,MALE,MD,armed-forces,exercise,own-child,0,0,21-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Columbus,6409 Cherokee Drive,21,1,NO,1,0,YES,76920,12820,6410,57690,Chevrolet,Malibu,2002,N
369,53,936638,20-05-1995,OH,250/500,2000,1459.93,0,438584,FEMALE,Masters,priv-house-serv,video-games,not-in-family,0,0,09-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Springfield,1123 5th Lane,10,2,YES,1,3,NO,77990,7090,14180,56720,Jeep,Wrangler,2012,N
261,46,348814,24-09-1992,IL,500/1000,1000,1245.61,0,478703,MALE,MD,transport-moving,base-jumping,own-child,0,0,12-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,5168 5th Ave,11,1,YES,1,0,?,59670,9180,9180,41310,Ford,Escape,2008,N
159,33,944102,20-07-2007,IN,100/300,2000,1462.76,0,615683,FEMALE,College,craft-repair,skydiving,husband,69200,-36900,24-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,NY,Columbus,3697 Apache Drive,23,3,YES,2,0,NO,44880,8160,4080,32640,Mercedes,C300,2004,Y
344,51,689901,28-04-1992,IN,100/300,2000,1398.46,0,455672,MALE,Associate,sales,skydiving,other-relative,0,0,02-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,NC,Northbend,1910 Sky Ave,14,1,?,0,2,NO,82830,7530,15060,60240,Audi,A5,2004,N
437,60,901083,19-01-1998,OH,500/1000,1000,1269.64,0,602942,FEMALE,College,armed-forces,cross-fit,unmarried,48800,0,14-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,SC,Riverwood,8954 Apache Lane,10,1,NO,1,3,NO,84480,7680,15360,61440,Chevrolet,Silverado,2012,Y
65,30,396224,08-09-2009,IN,100/300,500,1455.65,4000000,616706,FEMALE,College,transport-moving,skydiving,wife,0,-66300,15-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Hillsdale,3110 Lincoln Lane,6,3,?,2,3,NO,79800,15960,7980,55860,Honda,Civic,1999,N
280,41,682178,18-12-1994,OH,500/1000,2000,1140.31,0,473243,MALE,MD,adm-clerical,exercise,husband,29300,-64700,28-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,WV,Arlington,6035 Rock Ave,10,3,?,0,3,YES,53020,9640,9640,33740,Toyota,Corolla,1999,N
269,45,596298,23-08-1996,IN,500/1000,500,1330.46,0,435552,FEMALE,High School,machine-op-inspct,sleeping,wife,54800,-64100,18-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,VA,Hillsdale,2220 1st Lane,5,3,?,0,0,NO,24200,2200,4400,17600,Suburu,Forrestor,2008,N
275,40,253005,20-11-1991,OH,250/500,2000,1190.6,0,434206,MALE,Masters,exec-managerial,camping,unmarried,0,-45300,06-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,OH,Riverwood,4972 Francis Lane,17,1,?,1,3,YES,43230,7860,7860,27510,Chevrolet,Silverado,2001,N
265,45,985924,28-10-1998,OH,250/500,500,972.5,0,469895,FEMALE,College,exec-managerial,cross-fit,unmarried,0,0,19-01-2015,Vehicle Theft,?,Trivial Damage,None,NC,Springfield,6957 Weaver Drive,3,1,NO,2,3,NO,3190,290,580,2320,Ford,Escape,1995,N
283,43,631565,14-07-1997,IN,100/300,2000,1161.91,0,457722,FEMALE,Associate,adm-clerical,polo,not-in-family,0,-50400,17-01-2015,Parked Car,?,Minor Damage,None,WV,Northbrook,1512 Rock Lane,9,1,?,0,3,NO,5850,1300,650,3900,BMW,M5,2006,N
84,29,630998,09-04-2003,OH,250/500,1000,1117.17,0,473645,FEMALE,High School,machine-op-inspct,video-games,not-in-family,0,-29900,12-02-2015,Parked Car,?,Trivial Damage,Police,SC,Arlington,3693 Pine Ave,6,1,YES,2,0,YES,6820,620,1240,4960,BMW,3 Series,2005,N
247,44,926665,04-02-1992,OH,250/500,2000,1101.51,0,619108,FEMALE,College,farming-fishing,camping,not-in-family,64000,0,11-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Riverwood,9879 Apache Drive,22,3,NO,2,2,NO,69480,11580,11580,46320,Chevrolet,Tahoe,2008,N
56,29,302669,29-06-2006,IL,100/300,1000,1523.17,0,610479,MALE,Masters,prof-specialty,movies,own-child,0,0,21-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,PA,Northbend,2494 Andromedia Drive,10,3,?,1,2,YES,94560,7880,15760,70920,Jeep,Grand Cherokee,1995,N
210,39,620020,21-06-1997,OH,500/1000,1000,984.45,0,474998,MALE,Associate,armed-forces,paintball,unmarried,0,0,02-01-2015,Vehicle Theft,?,Minor Damage,None,NC,Riverwood,4615 Embaracadero Ave,4,1,YES,1,2,?,7800,780,780,6240,Dodge,RAM,1997,N
108,32,439828,07-09-2006,OH,500/1000,2000,1257,4000000,616341,FEMALE,High School,machine-op-inspct,basketball,unmarried,63900,-43700,11-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,WV,Northbrook,1929 Britain Drive,23,1,NO,1,3,NO,61270,5570,11140,44560,Suburu,Legacy,1999,N
328,49,971295,01-10-2001,OH,500/1000,500,1434.51,0,460535,FEMALE,Masters,transport-moving,bungie-jumping,wife,0,0,23-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,NY,Riverwood,5051 Elm St,19,1,?,0,2,YES,71440,8930,8930,53580,Mercedes,ML350,2005,N
186,37,165565,20-02-2009,OH,250/500,2000,1628,0,606487,FEMALE,JD,priv-house-serv,exercise,unmarried,0,0,28-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,NY,Hillsdale,9910 Maple Ave,22,1,YES,1,2,YES,55600,11120,5560,38920,Jeep,Grand Cherokee,2009,N
277,44,936543,26-06-2001,IN,500/1000,500,1412.31,0,620737,MALE,High School,priv-house-serv,board-games,unmarried,0,0,01-02-2015,Vehicle Theft,?,Trivial Damage,Police,NC,Riverwood,5602 Britain St,6,1,NO,1,3,NO,5000,1000,500,3500,Jeep,Wrangler,2005,N
138,33,296960,18-01-1997,IL,250/500,500,1362.87,5000000,445904,FEMALE,JD,exec-managerial,paintball,own-child,56900,-56900,24-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,NY,Springfield,6889 Cherokee St,6,1,NO,2,0,?,95810,14740,14740,66330,BMW,X5,2007,N
208,41,501692,24-06-2014,IN,100/300,1000,1134.68,0,464145,FEMALE,College,tech-support,chess,husband,0,0,20-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Northbend,3926 Rock Lane,18,1,NO,2,2,?,69300,6930,13860,48510,Volkswagen,Jetta,1996,N
147,37,525224,02-10-1992,IN,250/500,1000,1306.78,0,466818,MALE,MD,prof-specialty,video-games,other-relative,0,0,14-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Northbend,6717 Best Drive,22,1,?,1,0,NO,81120,13520,20280,47320,Toyota,Camry,1995,N
8,21,355085,09-10-2012,IN,500/1000,500,1021.9,0,464237,MALE,High School,handlers-cleaners,hiking,husband,0,0,05-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,WV,Columbus,6117 4th Ave,21,1,?,0,0,?,91260,14040,14040,63180,Toyota,Corolla,2012,N
297,48,830729,10-02-1993,IN,100/300,1000,1538.6,0,618455,FEMALE,MD,other-service,kayaking,wife,0,-54700,11-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,SC,Northbend,2668 Cherokee St,12,1,?,0,0,?,60600,6060,12120,42420,Ford,Fusion,2004,N
150,31,651948,28-09-1994,IN,500/1000,1000,1354.5,0,456602,MALE,Masters,machine-op-inspct,base-jumping,husband,52800,0,02-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Arlington,6838 Flute Lane,6,3,?,0,3,YES,64800,6480,12960,45360,Suburu,Forrestor,2000,Y
4,34,424358,24-05-2003,OH,500/1000,500,1282.93,0,616126,FEMALE,College,exec-managerial,basketball,other-relative,0,0,12-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,WV,Northbrook,6583 MLK Ridge,0,4,?,0,0,?,66880,6080,12160,48640,Chevrolet,Silverado,1996,Y
210,35,131478,25-12-1991,IL,500/1000,1000,1346.27,0,468508,MALE,Masters,farming-fishing,cross-fit,not-in-family,44900,-91400,03-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Arlington,6492 4th Lane,11,1,NO,0,2,?,58200,5820,5820,46560,BMW,X5,2013,N
91,31,268833,18-09-1999,IN,100/300,1000,1338.4,4000000,431937,FEMALE,High School,priv-house-serv,polo,own-child,63600,0,25-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,NC,Hillsdale,7299 Apache St,19,1,?,1,0,NO,60570,6730,6730,47110,Nissan,Maxima,2011,N
167,36,287489,03-02-1994,IL,100/300,1000,949.44,0,448603,FEMALE,Masters,exec-managerial,camping,other-relative,0,-38400,19-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,NC,Springfield,2756 Britain Hwy,22,3,?,0,0,NO,69680,8710,8710,52260,Mercedes,ML350,2008,Y
467,58,808153,18-01-2003,IN,500/1000,2000,977.4,0,444500,MALE,Masters,transport-moving,bungie-jumping,own-child,82200,0,03-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,9360 3rd Drive,2,1,YES,2,3,NO,55700,5570,11140,38990,Nissan,Maxima,2014,N
264,47,687639,07-03-2005,IN,250/500,2000,1181.46,10000000,601117,FEMALE,JD,transport-moving,sleeping,other-relative,0,-67400,12-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,OH,Arlington,1655 Francis Hwy,16,4,YES,1,2,YES,62370,5670,5670,51030,Dodge,Neon,2001,N
270,45,497347,23-08-2003,OH,500/1000,500,1187.53,0,615383,FEMALE,PhD,priv-house-serv,yachting,not-in-family,83200,-53300,24-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,VA,Columbus,9720 Lincoln Hwy,23,3,?,0,0,YES,54340,9880,0,44460,Saab,92x,2005,N
310,48,439660,11-07-2002,OH,100/300,1000,845.16,0,434342,FEMALE,Masters,priv-house-serv,base-jumping,other-relative,0,0,25-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NC,Northbend,7066 Texas Ave,21,1,YES,1,0,NO,55170,6130,6130,42910,Audi,A5,2005,Y
143,34,847123,19-03-2014,IL,100/300,500,1442.27,0,435100,MALE,College,priv-house-serv,exercise,wife,67900,0,17-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,NY,Columbus,9728 Britain Hwy,10,1,?,2,3,YES,58500,11700,5850,40950,Dodge,RAM,1999,N
146,32,172307,06-12-1993,OH,100/300,2000,1276.43,0,431278,MALE,Associate,priv-house-serv,skydiving,own-child,0,0,09-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NC,Riverwood,4486 Cherokee Ridge,12,1,?,0,3,?,59940,6660,6660,46620,Honda,CRV,1995,N
102,28,810189,29-08-1999,OH,250/500,500,1075.41,0,445648,MALE,MD,machine-op-inspct,reading,wife,55200,0,15-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,PA,Northbend,8021 Flute Ave,6,1,NO,1,0,NO,73400,7340,7340,58720,Dodge,Neon,1996,N
61,23,432068,09-03-2007,IL,100/300,500,1111.72,0,448857,MALE,JD,exec-managerial,bungie-jumping,other-relative,54600,0,25-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,PA,Riverwood,2774 Apache Drive,6,1,?,1,2,?,41850,4650,4650,32550,Suburu,Legacy,1997,N
255,44,903203,03-01-2004,OH,500/1000,2000,814.96,6000000,435267,FEMALE,PhD,priv-house-serv,chess,not-in-family,68500,0,05-02-2015,Parked Car,?,Trivial Damage,Police,NC,Hillsdale,2787 MLK St,7,1,?,2,2,NO,6400,640,1280,4480,Mercedes,ML350,2005,Y
211,40,253085,25-04-1991,IL,500/1000,1000,1575.86,0,461275,FEMALE,PhD,other-service,sleeping,own-child,0,0,12-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Northbrook,9847 Elm St,3,1,NO,1,1,NO,3190,580,290,2320,Audi,A5,2004,N
61,29,180720,14-03-1995,IN,250/500,1000,1115.27,0,613816,MALE,JD,handlers-cleaners,polo,unmarried,0,-66000,01-01-2015,Parked Car,?,Trivial Damage,None,VA,Hillsdale,4629 Elm Ridge,10,1,YES,2,1,YES,5900,590,590,4720,Nissan,Pathfinder,2010,N
108,31,492224,09-12-2005,IN,500/1000,2000,1175.7,0,608767,MALE,Masters,protective-serv,yachting,not-in-family,0,0,19-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,NY,Columbus,5585 Washington Drive,14,1,NO,0,2,NO,57330,6370,6370,44590,Dodge,Neon,2006,N
303,50,411477,25-12-2001,OH,100/300,500,793.15,0,620869,MALE,MD,tech-support,board-games,own-child,54600,-45500,14-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,WV,Hillsdale,3925 Sky St,17,3,NO,0,3,NO,81960,13660,13660,54640,Dodge,Neon,2008,N
152,33,107181,14-11-1999,IN,250/500,500,942.51,0,478981,FEMALE,PhD,transport-moving,exercise,wife,0,0,30-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,SC,Hillsdale,3903 Oak Ave,16,1,YES,0,0,?,70400,6400,19200,44800,Suburu,Legacy,2001,Y
120,34,312940,27-10-2001,IN,500/1000,1000,1056.71,0,464630,FEMALE,JD,protective-serv,paintball,not-in-family,77900,0,20-01-2015,Parked Car,?,Minor Damage,None,VA,Columbus,3805 Lincoln Hwy,3,1,NO,2,1,?,3770,580,580,2610,Jeep,Grand Cherokee,2002,N
144,36,855186,31-10-1993,IN,500/1000,2000,1255.68,6000000,466303,FEMALE,Associate,sales,reading,other-relative,23600,-15600,18-02-2015,Parked Car,?,Minor Damage,None,NC,Arlington,4055 2nd Drive,7,1,NO,0,0,?,7400,740,1480,5180,Dodge,Neon,2014,N
414,52,373935,13-02-2003,IN,500/1000,500,1335.13,0,452647,FEMALE,High School,farming-fishing,chess,unmarried,44000,-71000,07-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,SC,Northbend,3707 Oak Ridge,13,1,YES,1,1,?,54810,6090,6090,42630,Chevrolet,Silverado,1999,Y
163,37,812989,06-03-2004,IN,250/500,500,1178.95,6000000,441370,FEMALE,JD,priv-house-serv,dancing,own-child,0,-67300,16-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,NY,Springfield,7327 Lincoln Drive,20,1,YES,2,3,YES,49400,4940,9880,34580,Jeep,Wrangler,2005,N
352,53,993840,12-07-2013,IL,250/500,500,1793.16,0,619166,MALE,Associate,tech-support,exercise,wife,0,0,15-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NC,Riverwood,9369 Flute Hwy,23,1,YES,2,2,NO,68750,12500,6250,50000,Chevrolet,Malibu,2009,N
27,32,327856,27-08-2014,OH,100/300,500,1008.38,0,472803,FEMALE,PhD,adm-clerical,yachting,other-relative,37900,0,01-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Arlington,4239 Weaver Ave,11,1,NO,1,0,?,61500,12300,6150,43050,Dodge,Neon,2013,N
239,39,506333,22-06-1990,IL,100/300,500,1396.83,0,442308,FEMALE,Masters,other-service,reading,husband,0,0,06-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,WV,Northbend,6044 Weaver Drive,0,4,?,0,3,NO,76890,6990,13980,55920,BMW,X6,2007,N
33,32,263159,07-03-2008,OH,100/300,500,1402.78,5000000,469383,FEMALE,PhD,other-service,base-jumping,husband,70300,-50300,02-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Northbrook,8879 1st Drive,2,1,YES,0,1,?,56070,6230,12460,37380,Toyota,Camry,2012,N
88,30,372912,05-08-1992,IN,100/300,1000,1437.88,0,614383,FEMALE,College,transport-moving,reading,husband,42800,-51200,25-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,SC,Northbrook,9488 Best Drive,3,1,NO,2,0,YES,56000,14000,0,42000,Chevrolet,Malibu,2003,N
101,33,552788,03-09-1991,IL,500/1000,1000,1313.64,0,438617,FEMALE,College,priv-house-serv,board-games,unmarried,12100,0,10-02-2015,Parked Car,?,Trivial Damage,None,NY,Hillsdale,7500 Texas Ridge,3,1,YES,1,0,NO,4290,780,390,3120,Audi,A3,1997,N
20,37,722747,02-09-2011,IL,250/500,500,1482.14,0,613936,FEMALE,Associate,transport-moving,reading,husband,33000,-43600,22-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,NY,Arlington,2048 3rd Ridge,15,3,?,2,1,NO,60750,6750,6750,47250,Suburu,Forrestor,2003,N
126,30,248467,06-10-2012,IL,250/500,2000,1171.75,0,472163,FEMALE,Associate,machine-op-inspct,board-games,not-in-family,46500,-42700,31-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,NY,Northbend,3419 Apache St,23,1,?,2,3,NO,48730,4430,4430,39870,Chevrolet,Malibu,2011,N
264,43,955953,18-01-2014,IL,500/1000,2000,1353.33,0,447458,MALE,Associate,adm-clerical,video-games,not-in-family,0,-8500,12-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,WV,Springfield,9875 MLK Ave,9,3,YES,2,0,NO,95150,17300,17300,60550,Ford,Escape,2007,N
78,24,910622,22-03-1992,IN,100/300,500,1175.51,0,474792,MALE,Masters,craft-repair,yachting,husband,0,0,06-01-2015,Vehicle Theft,?,Minor Damage,Police,NY,Columbus,3553 Texas Ave,20,1,?,0,1,YES,7480,680,680,6120,Dodge,Neon,2003,N
123,28,137675,03-12-2012,IL,100/300,2000,1836.02,0,470559,MALE,Masters,transport-moving,movies,own-child,38000,-41200,01-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,NC,Riverwood,4335 1st St,5,1,?,2,1,YES,79800,13300,6650,59850,Volkswagen,Passat,2011,Y
347,51,343421,18-10-1996,OH,500/1000,500,1480.79,9000000,432399,FEMALE,MD,priv-house-serv,board-games,unmarried,0,-12100,07-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NC,Hillsdale,9070 Tree Ave,15,1,YES,0,2,?,103560,8630,17260,77670,Jeep,Wrangler,1997,N
163,38,413192,02-10-1997,IN,500/1000,2000,1453.92,0,607605,FEMALE,PhD,other-service,yachting,other-relative,51700,0,21-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Hillsdale,3900 Texas St,13,1,YES,2,3,?,79500,15900,7950,55650,Accura,MDX,1995,N
271,44,247801,18-03-2008,OH,250/500,500,1340.71,0,600153,FEMALE,High School,tech-support,base-jumping,other-relative,0,0,14-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,SC,Riverwood,9657 5th Ave,16,1,NO,2,2,NO,76230,6930,13860,55440,Volkswagen,Passat,1997,Y
410,54,171147,29-08-2010,IL,100/300,2000,714.03,0,465979,MALE,MD,protective-serv,board-games,own-child,0,-17000,16-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Northbend,5765 Washington St,0,1,YES,0,2,NO,59520,9920,9920,39680,Honda,Accord,2001,N
448,57,431283,31-03-2005,IL,100/300,2000,1376.16,0,466555,FEMALE,PhD,tech-support,hiking,own-child,38600,-50300,04-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Police,SC,Arlington,5997 Embaracadero Drive,10,3,NO,2,1,?,47760,5970,5970,35820,Suburu,Impreza,2013,Y
218,41,461962,25-12-2013,IL,100/300,500,914.22,0,444155,MALE,JD,prof-specialty,golf,wife,37900,-72900,22-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,NY,Springfield,1738 Solo Lane,14,3,NO,2,0,NO,84590,15380,7690,61520,Saab,93,2013,N
43,38,149467,11-03-2014,OH,500/1000,1000,1601.47,0,465764,MALE,PhD,handlers-cleaners,skydiving,other-relative,64400,0,15-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,SC,Columbus,2903 Weaver Drive,1,3,YES,2,2,?,61650,6850,6850,47950,Nissan,Ultima,2006,N
33,33,758740,04-08-1997,IL,500/1000,1000,1096.79,6000000,446898,FEMALE,Associate,handlers-cleaners,dancing,unmarried,45500,-60600,07-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,VA,Springfield,8926 Texas Ridge,16,1,?,2,1,?,81400,8140,8140,65120,BMW,M5,1998,N
126,34,628337,14-11-2007,IN,100/300,2000,1078.22,0,453274,FEMALE,Masters,transport-moving,camping,wife,54500,0,23-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,SC,Northbend,4231 3rd Ave,2,3,NO,0,1,YES,58410,10620,5310,42480,Chevrolet,Tahoe,2007,N
411,56,574637,30-07-1992,IL,250/500,1000,1595.28,0,479320,FEMALE,College,protective-serv,exercise,other-relative,0,0,06-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,VA,Springfield,8049 4th St,23,3,?,0,0,?,38610,3510,3510,31590,Volkswagen,Passat,2007,N
225,37,373600,01-12-2000,OH,100/300,1000,1217.84,5000000,443462,FEMALE,High School,farming-fishing,yachting,not-in-family,49600,0,07-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbrook,6501 5th Drive,19,3,?,2,2,NO,57600,9600,9600,38400,Volkswagen,Passat,2015,N
35,35,930032,10-09-2002,IL,100/300,2000,1117.42,0,446158,FEMALE,PhD,protective-serv,kayaking,not-in-family,0,-51900,14-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,NC,Hillsdale,7909 Andromedia Hwy,23,3,NO,2,2,NO,53190,5910,11820,35460,Volkswagen,Jetta,1996,N
460,57,396590,07-11-1997,OH,100/300,2000,1567.37,0,602514,FEMALE,JD,craft-repair,skydiving,husband,62500,0,15-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Riverwood,5865 Sky Lane,10,1,?,2,3,NO,58300,5830,11660,40810,Nissan,Maxima,2014,Y
195,38,238412,18-05-1993,IL,500/1000,2000,1294.93,6000000,477356,MALE,MD,tech-support,video-games,unmarried,38000,-50300,14-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,WV,Springfield,1957 Washington Ave,12,3,YES,1,2,NO,64620,7180,0,57440,Dodge,Neon,2003,N
360,51,484321,11-07-1996,IL,250/500,1000,1152.12,0,434669,MALE,PhD,armed-forces,hiking,not-in-family,0,-62400,05-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,WV,Riverwood,7649 Texas St,15,3,NO,2,0,YES,90480,15080,15080,60320,BMW,X6,2000,N
300,49,795847,17-12-1995,IL,100/300,1000,1441.21,0,609322,FEMALE,PhD,farming-fishing,kayaking,wife,0,0,04-01-2015,Vehicle Theft,?,Trivial Damage,None,SC,Columbus,1992 Britain Drive,3,1,NO,1,0,NO,7080,1180,1180,4720,Nissan,Maxima,2001,N
245,42,218456,16-07-2002,IL,500/1000,1000,1575.74,7000000,614265,MALE,JD,exec-managerial,chess,other-relative,0,-68900,20-02-2015,Parked Car,?,Minor Damage,Police,PA,Springfield,9685 Sky Ridge,19,1,?,2,2,?,6490,590,1180,4720,Toyota,Camry,2011,Y
146,36,792673,12-04-2013,OH,500/1000,2000,1233.96,0,606177,FEMALE,Masters,other-service,golf,other-relative,34500,-60600,05-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,NC,Hillsdale,3457 Texas Lane,16,3,NO,1,1,NO,55900,5590,5590,44720,Nissan,Ultima,2015,N
67,29,662256,13-11-1995,IL,250/500,1000,1861.43,0,461514,MALE,High School,adm-clerical,polo,husband,60400,-67800,15-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,SC,Columbus,7693 Cherokee Lane,16,1,NO,0,3,YES,63800,6380,6380,51040,Saab,92x,1998,Y
380,56,971338,04-11-2004,OH,100/300,1000,1570.86,0,454685,MALE,MD,farming-fishing,cross-fit,other-relative,66000,0,05-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,VA,Northbend,3167 4th Ridge,10,1,?,0,1,YES,58160,7270,7270,43620,Saab,95,1996,Y
389,53,714738,21-03-1998,IL,500/1000,2000,791.47,0,477260,MALE,Masters,armed-forces,chess,unmarried,0,0,01-01-2015,Vehicle Theft,?,Minor Damage,Police,NC,Riverwood,1617 Rock Drive,6,1,NO,1,2,NO,6300,630,1260,4410,Mercedes,C300,2001,Y
317,46,753844,22-07-1999,IN,250/500,1000,1012.78,0,469126,MALE,MD,sales,yachting,other-relative,43700,0,26-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,WV,Arlington,7877 3rd Ridge,18,1,NO,2,1,NO,104610,19020,19020,66570,Mercedes,ML350,1999,Y
264,44,976645,28-02-2010,IL,100/300,500,1047.06,6000000,443402,MALE,College,exec-managerial,sleeping,wife,0,0,13-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Springfield,9325 Lincoln Drive,20,3,NO,0,2,NO,69850,12700,6350,50800,Ford,Fusion,1999,N
20,21,918037,30-01-2005,OH,250/500,1000,1390.29,0,479408,FEMALE,Masters,priv-house-serv,polo,other-relative,0,-41200,14-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,PA,Northbrook,5855 Apache St,14,1,NO,0,0,?,62900,12580,6290,44030,Accura,MDX,2006,N
116,30,996253,29-11-2001,IN,500/1000,500,951.46,0,467227,MALE,JD,handlers-cleaners,golf,not-in-family,0,-35500,31-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,WV,Riverwood,1328 Texas Lane,8,3,NO,0,3,?,59670,6630,6630,46410,Volkswagen,Passat,2004,Y
222,40,373731,24-12-2012,IL,100/300,1000,1226.78,0,468433,MALE,JD,armed-forces,camping,unmarried,49600,-49200,22-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Springfield,4567 Pine Ave,17,2,YES,2,0,?,81500,16300,8150,57050,BMW,3 Series,2013,N
439,56,836272,11-05-1997,OH,100/300,500,1280.9,0,604289,MALE,High School,armed-forces,video-games,own-child,48900,-40900,30-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NC,Springfield,7575 Pine St,9,1,NO,2,0,?,50000,15000,5000,30000,Jeep,Wrangler,2009,N
66,28,167231,26-01-1994,IN,100/300,2000,1472.77,0,471366,MALE,Associate,adm-clerical,exercise,husband,0,-31700,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NY,Springfield,2850 Washington St,0,1,?,2,1,?,48290,8780,8780,30730,Nissan,Maxima,1995,N
128,29,743330,04-11-2010,OH,500/1000,1000,1878.44,0,450746,MALE,High School,other-service,golf,husband,0,-76000,20-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,VA,Riverwood,9169 Cherokee Hwy,2,1,NO,0,0,?,59070,5370,5370,48330,Chevrolet,Malibu,2003,N
69,24,807369,19-06-1992,IN,500/1000,500,1418.5,0,614948,FEMALE,High School,armed-forces,yachting,other-relative,0,0,04-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Springfield,6443 Washington Ridge,23,3,?,0,0,NO,63300,12660,6330,44310,Dodge,RAM,2012,N
294,46,735307,02-06-2010,IL,100/300,500,1532.8,0,473935,MALE,College,prof-specialty,exercise,own-child,0,0,23-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,SC,Northbend,6751 5th Hwy,8,1,NO,0,3,YES,65780,11960,11960,41860,Mercedes,ML350,2013,N
19,29,789208,12-10-2002,OH,250/500,500,1304.35,0,617267,MALE,JD,transport-moving,cross-fit,not-in-family,0,0,08-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,NY,Columbus,2289 Weaver Ridge,6,3,NO,0,2,?,75400,11600,11600,52200,Dodge,Neon,2005,Y
191,33,585324,25-02-2008,OH,500/1000,2000,1551.61,0,470670,MALE,High School,armed-forces,movies,unmarried,45000,-30400,21-02-2015,Vehicle Theft,?,Minor Damage,None,WV,Arlington,8306 1st Drive,3,1,YES,1,1,NO,2250,250,250,1750,Toyota,Corolla,2005,N
4,39,498759,05-09-1996,IL,100/300,1000,1326.98,6000000,450368,FEMALE,High School,machine-op-inspct,reading,unmarried,64200,0,13-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,VA,Northbend,2603 Andromedia Hwy,14,3,YES,1,3,?,54120,4510,9020,40590,Jeep,Grand Cherokee,2007,N
298,49,795004,16-03-1998,OH,250/500,500,862.92,0,448809,MALE,MD,machine-op-inspct,camping,wife,0,-71700,17-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,NY,Northbrook,6479 Francis Ave,16,3,NO,0,2,NO,69480,11580,11580,46320,Saab,95,2007,N
231,43,203250,22-04-2010,IN,100/300,2000,1331.69,0,469653,FEMALE,Masters,adm-clerical,reading,not-in-family,0,0,18-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,NY,Northbend,6428 Andromedia Lane,12,1,?,1,2,NO,66950,10300,10300,46350,Chevrolet,Malibu,2015,N
338,47,430794,25-01-2008,OH,250/500,2000,1486.04,0,615688,FEMALE,Associate,armed-forces,board-games,own-child,0,-56200,14-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,OH,Northbrook,9081 Cherokee Hwy,1,3,NO,2,3,?,64100,12820,6410,44870,Dodge,RAM,2014,Y
261,46,156636,10-09-2000,IN,100/300,1000,870.55,0,465631,MALE,PhD,prof-specialty,camping,unmarried,0,-49400,27-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,SC,Springfield,1532 Washington St,19,1,?,0,3,?,80280,13380,13380,53520,Chevrolet,Tahoe,2013,N
321,44,284143,23-04-2008,IL,500/1000,2000,1344.56,6000000,443344,MALE,Associate,machine-op-inspct,hiking,husband,0,-39100,12-02-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Northbend,4625 MLK Drive,7,1,?,1,2,NO,4680,520,0,4160,Accura,MDX,1999,N
0,32,740518,18-02-2011,OH,500/1000,1000,1377.04,0,441363,MALE,College,tech-support,base-jumping,wife,61400,-41100,17-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Springfield,1529 Elm Ridge,6,4,?,1,1,NO,39720,6620,6620,26480,Accura,MDX,2002,N
405,58,445289,24-04-2012,IL,250/500,500,1237.88,0,462683,MALE,MD,exec-managerial,exercise,not-in-family,0,-46900,13-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NY,Arlington,2086 Francis Drive,11,1,?,0,0,?,63580,5780,5780,52020,Mercedes,ML350,1997,Y
304,49,599262,25-09-2001,IN,100/300,1000,1525.86,0,463184,FEMALE,PhD,craft-repair,camping,own-child,0,0,21-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,NC,Northbend,9066 Best Ridge,2,1,YES,1,1,YES,73370,13340,6670,53360,Saab,95,2013,N
1,29,357949,24-05-2006,OH,500/1000,500,854.58,0,612826,FEMALE,JD,craft-repair,paintball,other-relative,52200,0,01-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Northbrook,7178 Best Drive,15,1,?,2,3,YES,86790,7890,23670,55230,Honda,CRV,2003,N
26,39,493161,30-01-1992,IN,250/500,1000,770.76,0,433155,MALE,Masters,tech-support,sleeping,husband,0,-53700,18-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,WV,Columbus,9821 Francis Ave,0,1,NO,0,2,?,49800,9960,4980,34860,Mercedes,ML350,2015,N
202,38,320251,24-01-2009,IL,100/300,2000,1132.74,0,616120,FEMALE,Associate,armed-forces,exercise,husband,0,-37500,04-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,WV,Hillsdale,7061 Cherokee Drive,12,1,YES,1,1,NO,77440,7040,14080,56320,Nissan,Ultima,2005,N
289,48,231127,29-08-1995,IL,500/1000,500,1173.37,8000000,461744,FEMALE,PhD,handlers-cleaners,board-games,own-child,0,-42700,09-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,SC,Springfield,1325 1st Lane,1,1,?,1,0,?,42900,8580,0,34320,Accura,TL,1999,N
61,26,766193,31-07-2011,OH,100/300,2000,1188.28,6000000,475916,FEMALE,JD,farming-fishing,skydiving,wife,0,-53800,14-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,VA,Hillsdale,3769 Sky St,16,2,YES,1,0,NO,53820,11960,5980,35880,Ford,F150,2015,Y
334,46,555374,05-01-2013,IL,100/300,1000,876.88,6000000,454434,MALE,MD,sales,reading,other-relative,0,0,03-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,WV,Hillsdale,8489 Pine Hwy,2,1,NO,2,1,NO,57330,12740,6370,38220,Jeep,Grand Cherokee,1998,N
12,24,491484,18-11-1994,IL,500/1000,1000,1143.95,0,464353,FEMALE,PhD,tech-support,paintball,other-relative,51400,0,04-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Riverwood,6329 Apache Ave,13,3,NO,2,1,?,53370,5930,5930,41510,Nissan,Ultima,2011,N
86,29,925128,30-08-2014,IL,100/300,2000,1409.06,0,610302,MALE,High School,prof-specialty,yachting,husband,74200,-68100,30-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Northbrook,9293 Pine Lane,0,1,NO,2,2,YES,62920,9680,14520,38720,Accura,MDX,2005,N
83,24,265093,01-01-2006,IN,500/1000,1000,1070.63,0,462106,FEMALE,High School,machine-op-inspct,board-games,unmarried,0,0,20-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,NY,Arlington,9224 Sky Drive,0,3,?,0,1,NO,61600,6160,12320,43120,Honda,CRV,2003,N
126,30,267808,10-09-1998,IL,500/1000,2000,916.13,0,431389,MALE,College,sales,golf,unmarried,55300,-58400,07-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,NY,Northbend,8862 Maple Ridge,16,3,YES,2,0,NO,74160,6180,12360,55620,Mercedes,E400,2002,N
209,38,116735,28-01-2010,OH,250/500,500,1191.5,0,442866,MALE,High School,priv-house-serv,reading,husband,38600,-52900,31-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,WV,Arlington,3492 Flute Lane,8,1,NO,2,0,?,80100,8900,8900,62300,Audi,A3,2015,N
283,48,963680,04-01-2003,OH,500/1000,1000,1474.66,0,446755,FEMALE,JD,sales,paintball,husband,0,-46200,17-02-2015,Parked Car,?,Trivial Damage,Police,NY,Hillsdale,6484 Tree Drive,9,1,?,2,3,NO,6560,820,820,4920,Volkswagen,Jetta,2003,N
194,34,445694,24-05-2004,IL,250/500,1000,1193.45,0,464743,MALE,JD,other-service,hiking,not-in-family,0,0,24-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Northbend,4554 Sky Ave,11,1,?,2,1,YES,58800,11760,5880,41160,Nissan,Pathfinder,1997,N
184,38,215534,12-09-1994,IL,250/500,1000,1437.53,0,437889,FEMALE,College,transport-moving,chess,not-in-family,0,0,02-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,PA,Northbrook,5201 Texas Hwy,6,3,?,0,2,NO,53730,11940,5970,35820,Dodge,RAM,2013,Y
479,60,232854,07-07-1997,IL,100/300,2000,1304.83,0,473638,FEMALE,College,other-service,cross-fit,husband,0,0,09-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,NY,Arlington,3982 Weaver Lane,18,1,NO,0,0,NO,60600,5050,10100,45450,Honda,Civic,2001,N
284,48,168260,01-03-1991,OH,250/500,1000,1168.8,0,444232,FEMALE,JD,tech-support,movies,other-relative,0,-42400,28-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Northbrook,3660 Andromedia Hwy,11,1,?,0,3,YES,35750,6500,3250,26000,Mercedes,E400,2001,N
65,27,538955,29-09-2001,IN,100/300,1000,1164.97,0,477695,FEMALE,College,adm-clerical,exercise,wife,43000,-42500,17-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Arlington,7135 Flute Lane,17,1,?,1,2,YES,42840,3570,7140,32130,Chevrolet,Silverado,2004,N
222,39,243226,10-01-2012,IL,250/500,1000,1232.72,0,458237,MALE,High School,armed-forces,hiking,own-child,87800,-51200,09-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Springfield,4414 Solo Drive,21,3,?,1,0,NO,87960,14660,14660,58640,Jeep,Wrangler,1999,Y
196,41,246435,05-07-2001,IL,250/500,2000,1800.76,0,441499,MALE,JD,protective-serv,camping,other-relative,0,-78600,14-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,SC,Hillsdale,2920 5th Ave,0,3,NO,1,0,NO,47800,4780,4780,38240,Jeep,Grand Cherokee,2009,N
253,43,582480,07-08-1991,IL,500/1000,500,1187.01,7000000,613436,FEMALE,Associate,tech-support,exercise,unmarried,46300,-33000,02-02-2015,Vehicle Theft,?,Trivial Damage,Police,NY,Northbrook,2986 MLK Drive,9,1,NO,0,1,?,3840,640,640,2560,Chevrolet,Tahoe,2014,N
280,43,345539,24-07-2012,IN,100/300,1000,1559.34,0,448912,MALE,JD,transport-moving,hiking,own-child,0,-51600,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Riverwood,1580 Maple Lane,1,1,NO,0,2,?,77000,14000,7000,56000,Accura,RSX,2004,N
5,26,924318,27-07-2014,IL,250/500,2000,1137.02,0,468872,FEMALE,PhD,farming-fishing,skydiving,not-in-family,31500,0,25-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Springfield,3706 Texas Hwy,22,1,YES,1,3,?,88110,16020,16020,56070,Audi,A5,2003,N
220,42,726880,08-08-1994,IN,100/300,1000,1281.72,0,619811,MALE,College,farming-fishing,hiking,other-relative,33500,-49500,13-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Northbend,9109 Britain Drive,20,4,NO,0,2,YES,47740,4340,4340,39060,Honda,Civic,2005,N
85,30,190588,09-12-2001,OH,100/300,1000,796.35,0,614166,FEMALE,MD,craft-repair,video-games,own-child,72400,-77000,20-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Northbend,2290 4th Ave,9,3,YES,2,1,YES,58960,5360,10720,42880,Ford,F150,2004,N
266,46,246705,14-03-1990,OH,250/500,500,1270.02,0,456600,FEMALE,Associate,tech-support,skydiving,own-child,0,-45800,08-01-2015,Parked Car,?,Minor Damage,Police,NC,Northbrook,4232 Britain Ridge,5,1,NO,1,2,?,2160,480,240,1440,Toyota,Corolla,2004,N
41,26,619589,28-03-2006,IL,100/300,1000,1383.13,0,618405,FEMALE,JD,prof-specialty,exercise,own-child,46700,0,28-02-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Riverwood,6677 Andromedia Drive,12,1,YES,1,1,?,6890,530,1060,5300,Jeep,Grand Cherokee,1997,N
316,45,164988,23-12-2013,IL,100/300,2000,1290.74,5000000,430832,FEMALE,High School,prof-specialty,kayaking,husband,58300,0,12-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,VA,Springfield,5868 Sky Hwy,6,3,NO,2,0,YES,78870,7170,14340,57360,Suburu,Legacy,2013,N
285,47,729534,30-09-1991,IN,100/300,1000,1216.68,0,610989,FEMALE,Masters,sales,basketball,other-relative,55100,0,06-01-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Columbus,3053 Lincoln Drive,8,1,NO,1,1,NO,2700,300,300,2100,Ford,F150,2013,N
379,54,505014,27-12-2001,IL,100/300,500,1251.16,0,447750,FEMALE,Associate,machine-op-inspct,kayaking,not-in-family,41400,0,15-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Riverwood,7041 Tree Ridge,14,1,?,0,1,NO,75960,6330,6330,63300,Jeep,Grand Cherokee,2010,N
15,34,920826,07-04-2005,IN,250/500,2000,1586.41,0,608708,FEMALE,High School,sales,video-games,other-relative,33500,-58900,20-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,WV,Northbrook,7223 Embaracadero St,10,1,YES,1,3,?,75570,6870,13740,54960,BMW,X5,2010,Y
354,48,534982,08-04-2003,IL,500/1000,2000,1526.11,5000000,469650,FEMALE,Masters,sales,exercise,unmarried,0,0,03-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,SC,Columbus,8081 Flute Ridge,12,1,?,2,3,YES,90240,15040,15040,60160,Chevrolet,Malibu,1995,N
342,53,110408,14-11-2005,IN,100/300,1000,1028.44,0,602304,FEMALE,College,prof-specialty,dancing,not-in-family,0,0,26-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,SC,Springfield,8618 Texas Lane,12,1,?,0,0,NO,80960,14720,7360,58880,Accura,MDX,2000,N
169,38,283052,07-01-2005,IL,100/300,1000,1555.94,0,459878,MALE,PhD,craft-repair,skydiving,own-child,23300,0,25-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,VA,Riverwood,3508 Washington St,12,3,NO,1,3,YES,79080,6590,13180,59310,Mercedes,C300,2012,N
339,49,840806,14-02-1994,IN,500/1000,2000,1570.77,0,441142,MALE,JD,adm-clerical,paintball,not-in-family,98800,-65300,18-01-2015,Vehicle Theft,?,Minor Damage,None,SC,Columbus,2193 4th Ridge,13,1,NO,0,3,NO,6820,1240,620,4960,Mercedes,ML350,2009,N
259,42,382394,23-01-1996,OH,100/300,2000,1170.53,0,465667,FEMALE,PhD,armed-forces,sleeping,wife,65000,-49200,12-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,WV,Northbend,8897 Sky St,17,3,NO,1,2,YES,62590,5690,11380,45520,Nissan,Pathfinder,2006,N
65,23,876699,12-12-1999,OH,250/500,1000,1099.95,0,473109,FEMALE,College,sales,dancing,wife,0,-71900,15-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NY,Arlington,9611 Pine Ridge,14,1,NO,1,0,YES,52400,6550,6550,39300,Accura,MDX,2005,Y
254,46,871432,15-07-2004,IL,250/500,2000,1472.43,0,619794,MALE,MD,tech-support,basketball,husband,0,-90600,10-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,WV,Arlington,7825 1st Ridge,3,3,?,1,3,YES,63580,5780,11560,46240,Volkswagen,Jetta,2004,N
440,55,379882,07-11-2012,IL,250/500,500,1275.62,0,602258,FEMALE,Associate,priv-house-serv,reading,other-relative,0,-56200,23-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,NY,Riverwood,3039 Oak Hwy,18,3,YES,2,1,NO,61400,6140,6140,49120,Nissan,Ultima,1995,N
478,63,852002,29-06-2009,IL,250/500,1000,1292.3,0,479724,MALE,High School,adm-clerical,paintball,own-child,47600,0,21-02-2015,Parked Car,?,Minor Damage,None,VA,Northbend,8204 Pine Lane,5,1,YES,1,3,NO,4700,940,470,3290,Dodge,Neon,2007,Y
230,44,372891,26-06-2000,IN,250/500,2000,1009.37,0,442210,FEMALE,College,prof-specialty,hiking,other-relative,45400,-39400,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,SC,Riverwood,9787 Andromedia Ave,19,1,NO,0,2,YES,74140,13480,13480,47180,BMW,X5,2015,N
138,30,689034,09-01-2002,OH,500/1000,500,1093.07,4000000,463291,FEMALE,PhD,other-service,reading,wife,27700,-72400,06-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,VA,Hillsdale,9633 Rock Hwy,0,1,?,2,2,NO,83160,6930,13860,62370,Volkswagen,Jetta,2011,N
239,41,743092,11-11-2013,OH,250/500,1000,1325.44,7000000,474898,FEMALE,JD,farming-fishing,paintball,other-relative,51400,-6300,18-02-2015,Parked Car,?,Trivial Damage,Police,NC,Arlington,6303 1st Drive,22,1,?,0,2,YES,10790,1660,830,8300,Mercedes,E400,2013,N
93,31,599174,14-01-2008,IL,100/300,2000,1017.18,0,431354,FEMALE,MD,prof-specialty,paintball,husband,0,0,17-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NC,Arlington,2014 Rock Ave,21,3,YES,1,3,NO,48070,8740,8740,30590,Saab,92x,2014,N
37,25,421092,04-03-2003,OH,100/300,1000,1221.17,0,617460,FEMALE,Masters,protective-serv,golf,not-in-family,49300,0,24-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,SC,Northbrook,8983 Tree St,4,3,YES,0,0,YES,51030,5670,11340,34020,Suburu,Impreza,1996,N
254,40,349658,07-06-1994,IN,100/300,500,1927.87,0,609317,MALE,MD,prof-specialty,yachting,husband,0,0,21-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,VA,Arlington,6260 5th Lane,10,1,YES,0,1,?,43280,0,5410,37870,Honda,Civic,1996,Y
131,29,811042,04-07-2013,IN,250/500,1000,978.27,0,479821,FEMALE,Associate,sales,paintball,own-child,65700,0,03-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Hillsdale,2725 Britain Ridge,5,1,?,1,3,NO,76400,15280,7640,53480,Suburu,Forrestor,2003,N
230,43,505316,30-06-2002,IN,100/300,2000,1221.14,0,473394,MALE,MD,prof-specialty,board-games,wife,48100,0,07-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,VA,Hillsdale,3089 Oak Ridge,13,1,?,0,2,?,75460,13720,13720,48020,Audi,A5,2002,N
313,50,116645,30-06-2004,OH,100/300,2000,1255.62,0,603882,MALE,MD,armed-forces,polo,unmarried,0,0,02-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,OH,Northbrook,6206 3rd Ridge,18,3,YES,2,1,YES,69000,13800,6900,48300,Ford,F150,1995,Y
210,38,950880,19-12-1998,IN,250/500,500,999.52,0,615229,MALE,JD,tech-support,golf,other-relative,0,0,13-01-2015,Vehicle Theft,?,Minor Damage,Police,VA,Springfield,7240 5th Ridge,6,1,?,1,2,NO,8640,1440,720,6480,Accura,TL,2008,N
101,29,788502,31-08-2014,OH,250/500,500,1380.89,0,620197,MALE,PhD,armed-forces,dancing,own-child,30000,-53000,28-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,SC,Arlington,8100 3rd Ave,0,3,?,2,1,?,67210,12220,12220,42770,BMW,X6,1996,N
153,37,627486,10-11-2005,IN,500/1000,500,1010.77,0,438215,MALE,High School,transport-moving,basketball,unmarried,52300,-55600,16-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,NC,Arlington,3282 4th Lane,5,3,?,0,3,NO,42500,4250,4250,34000,Volkswagen,Jetta,1999,N
337,53,498842,04-05-2000,OH,100/300,500,1205.86,0,444583,MALE,Associate,armed-forces,basketball,wife,0,-34600,01-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Arlington,3227 Maple Ave,8,1,YES,0,1,NO,86400,14400,7200,64800,Accura,RSX,2001,N
360,51,550294,26-11-2001,IL,500/1000,1000,1526.61,0,471866,MALE,Masters,handlers-cleaners,chess,not-in-family,0,-32900,30-01-2015,Vehicle Theft,?,Minor Damage,Police,SC,Hillsdale,4264 Lincoln Ridge,5,1,YES,2,2,?,4620,840,840,2940,Dodge,RAM,2009,Y
428,53,328387,06-05-2014,IL,100/300,1000,1496.44,0,616884,FEMALE,High School,tech-support,camping,unmarried,0,0,16-02-2015,Parked Car,?,Trivial Damage,Police,NC,Springfield,2215 Best Ave,9,1,YES,2,2,NO,6930,630,1260,5040,Honda,CRV,2013,N
204,40,540152,27-01-1991,IL,100/300,500,1256.2,0,448310,FEMALE,JD,sales,hiking,not-in-family,0,0,07-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,SC,Northbend,5363 Weaver Lane,10,3,YES,1,2,YES,41700,8340,8340,25020,Saab,95,2013,N
364,51,385932,28-04-1992,IL,100/300,500,1268.35,0,478902,MALE,Masters,transport-moving,board-games,wife,0,0,10-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,NY,Northbrook,2397 Cherokee Ave,16,1,YES,2,1,NO,77330,14060,14060,49210,Volkswagen,Jetta,2014,Y
185,35,618682,04-03-2000,IN,500/1000,2000,1421.59,0,442695,MALE,College,other-service,sleeping,own-child,0,0,31-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Arlington,9794 Embaracadero St,8,1,?,2,3,YES,4950,900,450,3600,Ford,F150,2011,N
63,26,550930,12-10-1995,IL,500/1000,500,1500.04,6000000,613826,MALE,PhD,craft-repair,polo,own-child,0,-36500,13-02-2015,Vehicle Theft,?,Minor Damage,Police,NC,Northbrook,1810 Elm Hwy,5,1,NO,0,2,YES,5160,860,860,3440,Accura,TL,2004,N
210,35,998192,25-04-2014,IL,100/300,500,1433.24,0,476203,FEMALE,College,exec-managerial,yachting,not-in-family,0,-19500,22-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Riverwood,9603 Texas Lane,11,3,NO,2,1,?,24570,2730,2730,19110,Saab,95,2006,Y
194,38,663938,26-01-2011,IN,100/300,2000,1231.25,0,604333,FEMALE,PhD,craft-repair,movies,not-in-family,46500,0,08-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Arlington,5650 Sky Drive,15,3,?,1,0,?,53680,4880,9760,39040,Toyota,Camry,2011,N
294,49,756870,26-01-1996,IN,500/1000,500,1135.43,0,442604,MALE,Masters,farming-fishing,bungie-jumping,own-child,22700,0,04-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,WV,Columbus,9633 MLK Lane,23,3,YES,1,1,YES,42900,3900,3900,35100,Chevrolet,Malibu,2010,N
272,41,337158,08-04-1991,OH,250/500,2000,945.73,5000000,435663,MALE,MD,protective-serv,chess,wife,38600,-42800,04-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NY,Arlington,4981 Flute Hwy,23,1,NO,0,0,NO,84100,16820,8410,58870,Ford,Escape,2009,Y
27,27,919875,29-06-2002,IN,100/300,2000,1118.76,0,470866,FEMALE,College,adm-clerical,dancing,own-child,0,-55800,26-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Northbrook,9078 Francis Ridge,23,1,?,1,3,?,61560,6840,6840,47880,Ford,Fusion,2008,N
251,39,315631,09-04-1999,IN,500/1000,2000,1231.98,0,612908,FEMALE,Associate,other-service,hiking,not-in-family,0,-31700,08-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,NY,Hillsdale,1381 Francis Ave,10,1,YES,0,0,?,44240,5530,5530,33180,Dodge,RAM,1997,N
180,33,113464,19-04-2009,IN,500/1000,2000,1005.47,0,441871,FEMALE,JD,protective-serv,hiking,own-child,58100,-49000,15-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,VA,Columbus,6435 Texas Ave,12,4,?,2,3,YES,57700,11540,5770,40390,Jeep,Grand Cherokee,2002,N
392,50,556415,22-08-1991,OH,100/300,2000,1108.97,0,431496,FEMALE,PhD,exec-managerial,exercise,not-in-family,68400,-66800,14-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,WV,Springfield,1248 MLK Ridge,4,1,NO,2,2,YES,108030,16620,16620,74790,Saab,92x,2002,N
143,30,250249,28-11-1991,IN,100/300,500,1392.39,5000000,436499,FEMALE,High School,exec-managerial,dancing,unmarried,0,-65700,12-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Riverwood,3323 1st Lane,16,1,NO,2,0,YES,54300,10860,5430,38010,Toyota,Highlander,2010,N
371,54,403776,27-04-2012,IN,100/300,2000,1317.97,0,469853,MALE,High School,craft-repair,movies,wife,34700,-81000,18-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,SC,Columbus,6971 Best Ridge,18,3,?,1,2,?,32280,5380,5380,21520,Ford,Fusion,2010,Y
292,42,396002,04-03-2007,IN,250/500,1000,1588.22,0,605369,MALE,JD,machine-op-inspct,camping,other-relative,0,-53800,15-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,WV,Riverwood,7488 Lincoln Lane,15,3,YES,1,1,NO,84600,16920,8460,59220,Chevrolet,Malibu,2007,N
165,35,976908,31-12-2012,IL,250/500,500,900.02,6000000,448466,MALE,College,craft-repair,camping,own-child,0,-49900,24-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,SC,Springfield,9007 Francis Hwy,8,1,NO,1,3,YES,69700,6970,6970,55760,BMW,3 Series,2008,N
158,33,509489,21-12-2013,OH,100/300,1000,1744.64,3000000,432786,MALE,JD,prof-specialty,movies,unmarried,0,0,07-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Springfield,1491 Francis Ridge,4,3,NO,0,1,NO,36400,3640,7280,25480,Volkswagen,Jetta,1998,N
241,39,485295,28-04-2005,OH,250/500,1000,1260.56,0,473591,FEMALE,JD,adm-clerical,paintball,own-child,0,-54900,22-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,SC,Northbrook,3659 Oak Lane,20,3,NO,0,2,YES,37520,4690,4690,28140,Jeep,Wrangler,2010,N
103,33,361829,17-09-1994,OH,500/1000,2000,1021.14,0,618418,FEMALE,Masters,other-service,paintball,wife,69500,-47700,19-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Northbend,4176 Britain Hwy,1,3,NO,2,3,?,79090,14380,14380,50330,Dodge,RAM,2014,N
402,54,603632,16-08-2003,OH,250/500,2000,1285.09,0,444558,MALE,JD,farming-fishing,board-games,not-in-family,48000,-79600,28-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,NY,Springfield,5189 Francis Drive,19,1,NO,0,2,YES,67770,7530,15060,45180,Mercedes,ML350,2013,Y
102,32,783494,02-09-2014,OH,100/300,500,1537.07,3000000,457733,MALE,JD,tech-support,chess,wife,0,0,04-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,OH,Northbend,6515 Oak Lane,11,1,NO,1,0,NO,47400,9480,4740,33180,Chevrolet,Silverado,2004,Y
182,40,439049,12-12-2011,IN,100/300,1000,1022.42,0,466161,MALE,PhD,other-service,skydiving,husband,50000,-56900,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,SC,Northbend,7168 Andromedia Ridge,13,1,YES,0,2,?,71100,7110,14220,49770,Audi,A3,2008,N
282,46,502634,17-08-1991,OH,100/300,2000,1558.86,0,450800,MALE,MD,other-service,dancing,wife,51100,-75100,17-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Springfield,7954 Tree Ridge,2,1,?,2,2,NO,69400,13880,6940,48580,BMW,M5,2012,N
222,39,378588,29-02-2004,OH,500/1000,500,1757.87,0,458993,MALE,High School,transport-moving,board-games,not-in-family,71400,0,17-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Northbrook,1956 Apache St,9,3,YES,2,1,?,55000,5000,10000,40000,Saab,93,1996,Y
415,52,794731,22-02-2015,IN,250/500,1000,973.5,0,468634,MALE,PhD,machine-op-inspct,polo,not-in-family,50400,0,02-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,WV,Arlington,9918 Andromedia Drive,15,2,YES,1,3,YES,51090,7860,7860,35370,Toyota,Highlander,2003,N
51,34,641934,25-12-2013,OH,500/1000,500,1430.8,0,461264,MALE,PhD,machine-op-inspct,camping,unmarried,0,0,12-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,NY,Springfield,5499 Flute Ridge,23,1,?,2,3,NO,64200,6420,19260,38520,Honda,Civic,2007,Y
255,45,113516,13-10-1990,IL,500/1000,500,1192.27,0,600184,MALE,High School,sales,base-jumping,own-child,0,-40200,04-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NC,Riverwood,3311 2nd Drive,16,1,NO,2,0,YES,67320,12240,12240,42840,Ford,Fusion,2006,N
143,31,425631,05-07-2014,IL,250/500,500,1163.83,0,604874,MALE,Associate,protective-serv,movies,husband,37700,0,21-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,NC,Arlington,7609 Rock St,21,4,YES,2,0,?,76120,6920,13840,55360,Audi,A5,1999,N
130,28,542245,25-11-1991,OH,500/1000,1000,1003.15,0,462377,FEMALE,JD,farming-fishing,movies,other-relative,0,-38500,23-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,SC,Hillsdale,4652 Flute Drive,21,2,YES,2,1,NO,85020,13080,13080,58860,Ford,Fusion,2010,N
242,41,512894,02-10-1990,OH,250/500,2000,1153.54,6000000,619657,MALE,Masters,protective-serv,polo,unmarried,0,-57000,12-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Northbrook,6853 Sky Hwy,3,3,NO,0,1,?,68090,12380,12380,43330,Toyota,Corolla,2009,N
96,27,633090,17-02-2009,IL,100/300,1000,1631.1,0,437323,FEMALE,High School,priv-house-serv,exercise,wife,0,0,23-01-2015,Parked Car,?,Trivial Damage,Police,WV,Arlington,7780 Flute Lane,4,1,?,1,2,NO,6030,670,670,4690,Nissan,Pathfinder,2007,N
180,35,464234,17-07-2005,IL,500/1000,1000,1252.48,0,432148,MALE,MD,machine-op-inspct,yachting,wife,0,-55800,10-02-2015,Vehicle Theft,?,Minor Damage,None,OH,Springfield,1687 3rd Lane,17,1,?,1,3,NO,5100,1020,510,3570,Chevrolet,Malibu,2000,N
150,30,290162,12-03-1994,IN,100/300,1000,1677.26,0,439690,MALE,College,sales,yachting,own-child,40100,0,10-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Springfield,6378 Britain Ave,7,1,YES,1,3,YES,4590,510,510,3570,Volkswagen,Jetta,2013,N
463,59,638155,03-08-1994,IL,250/500,1000,979.73,0,601848,FEMALE,JD,exec-managerial,yachting,not-in-family,51700,0,12-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,SC,Columbus,1306 Andromedia St,14,2,?,1,2,NO,72400,7240,14480,50680,Chevrolet,Tahoe,1999,Y
472,64,911429,25-08-2012,IN,250/500,500,989.24,0,615821,MALE,Masters,other-service,skydiving,not-in-family,0,0,21-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,NY,Northbrook,3664 Francis Ridge,13,1,NO,2,3,NO,70900,14180,7090,49630,Mercedes,ML350,2002,N
75,25,106186,02-12-2011,IL,500/1000,1000,1389.86,0,472475,FEMALE,Associate,priv-house-serv,hiking,husband,0,0,18-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Springfield,5985 Lincoln Lane,23,2,?,2,3,YES,65100,6510,6510,52080,Saab,93,2011,N
193,40,311783,25-02-2005,OH,100/300,500,1233.85,0,457463,FEMALE,College,handlers-cleaners,chess,husband,0,0,28-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,PA,Hillsdale,3706 4th Hwy,23,3,?,2,1,YES,64260,0,14280,49980,Ford,Escape,1999,N
43,43,528385,07-11-1997,IL,500/1000,500,1320.39,0,604861,FEMALE,Associate,armed-forces,yachting,not-in-family,0,0,19-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,WV,Arlington,6603 Francis Hwy,16,1,?,2,1,?,79970,7270,21810,50890,Honda,CRV,1996,Y
253,41,228403,20-04-2004,IN,100/300,1000,1435.09,0,471519,FEMALE,College,machine-op-inspct,bungie-jumping,not-in-family,36600,0,11-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,SC,Northbrook,7069 4th Hwy,17,1,NO,2,0,?,56610,6290,6290,44030,Chevrolet,Tahoe,1995,N
152,30,209177,17-11-2009,IN,500/1000,500,1448.54,0,618682,FEMALE,JD,craft-repair,polo,wife,58600,0,11-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,VA,Northbrook,5093 Flute Lane,9,1,YES,1,1,?,84590,7690,7690,69210,Toyota,Highlander,2000,Y
160,38,497929,19-09-2009,OH,250/500,500,1733.56,0,441425,MALE,High School,sales,sleeping,wife,0,-43800,09-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,SC,Hillsdale,5894 Flute Drive,13,3,NO,2,1,YES,66780,7420,14840,44520,Mercedes,ML350,1996,N
56,36,735844,08-11-2009,IN,100/300,500,1533.07,0,609336,MALE,JD,farming-fishing,exercise,own-child,0,-28800,20-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,NC,Springfield,8459 Apache Ave,13,1,YES,1,2,YES,58500,0,6500,52000,Ford,Escape,2001,N
286,41,710741,12-09-2001,IL,100/300,500,1106.77,0,603320,FEMALE,College,prof-specialty,golf,other-relative,45500,-62500,26-02-2015,Parked Car,?,Trivial Damage,Police,WV,Columbus,7447 Lincoln Ridge,3,1,?,2,0,NO,5000,500,500,4000,Dodge,RAM,2003,N
3,29,276804,27-11-1992,IL,100/300,500,995.7,5000000,615446,FEMALE,JD,priv-house-serv,chess,unmarried,0,0,02-02-2015,Parked Car,?,Trivial Damage,Police,PA,Springfield,1821 Andromedia Ridge,3,1,?,2,1,?,5000,500,1000,3500,Mercedes,E400,2008,Y
286,41,507545,07-12-1998,IL,250/500,1000,1298.85,6000000,435967,FEMALE,High School,sales,camping,other-relative,71300,-70300,04-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Columbus,6859 Flute Ridge,16,3,?,1,3,YES,54450,6050,12100,36300,Saab,92x,2007,N
239,38,485642,25-08-1990,OH,250/500,1000,1276.73,5000000,610246,FEMALE,Masters,handlers-cleaners,bungie-jumping,other-relative,0,0,19-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Police,WV,Northbrook,4175 Elm Ridge,12,3,NO,1,3,?,61920,6880,6880,48160,Saab,93,2003,N
64,29,796375,22-10-2011,OH,250/500,2000,1202.28,0,479327,MALE,High School,exec-managerial,cross-fit,other-relative,0,-61400,12-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,VA,Hillsdale,5007 Oak St,4,1,?,1,2,NO,43700,4370,4370,34960,Nissan,Pathfinder,2007,Y
98,31,171183,01-02-1990,IN,100/300,500,671.92,0,468300,MALE,Masters,machine-op-inspct,bungie-jumping,wife,0,-26400,24-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,NC,Riverwood,5790 Flute Ridge,3,1,?,2,0,?,64080,7120,7120,49840,Ford,Escape,1997,N
16,35,110084,27-11-1990,IL,250/500,1000,1358.03,0,612660,MALE,JD,sales,chess,own-child,59300,-31400,17-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,SC,Hillsdale,8704 Britain Lane,0,1,NO,2,1,?,55000,10000,10000,35000,Volkswagen,Jetta,2008,Y
70,27,714784,16-07-2004,IN,250/500,1000,1008.79,4000000,466691,FEMALE,Masters,adm-clerical,video-games,own-child,46000,0,13-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Columbus,7816 MLK Lane,19,1,NO,2,3,?,4400,0,550,3850,Toyota,Camry,2000,N
75,27,143924,10-12-1993,OH,100/300,1000,1141.1,0,468515,MALE,JD,armed-forces,movies,other-relative,0,0,28-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,SC,Hillsdale,3618 Maple Lane,15,2,?,0,1,YES,71640,5970,11940,53730,Toyota,Highlander,2008,N
246,44,996850,08-03-1995,OH,100/300,1000,1397,0,614521,MALE,High School,machine-op-inspct,reading,not-in-family,0,0,03-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,NY,Arlington,7705 Lincoln Drive,6,1,NO,1,0,NO,61740,6860,6860,48020,Accura,MDX,1997,N
110,27,284834,03-08-2009,OH,500/1000,1000,1664.66,0,465921,FEMALE,Associate,priv-house-serv,golf,own-child,0,-66200,05-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,SC,Springfield,8602 Washington Ridge,3,1,?,0,3,?,57500,5750,5750,46000,Audi,A3,2010,N
236,39,830878,03-11-1996,IN,250/500,1000,1151.39,4000000,604555,FEMALE,Masters,exec-managerial,reading,wife,0,-63900,01-01-2015,Parked Car,?,Minor Damage,Police,NY,Springfield,2832 Andromedia Lane,17,1,YES,0,0,?,8700,870,1740,6090,Accura,RSX,2003,N
267,46,270208,09-08-2004,OH,100/300,2000,1546.01,0,616276,FEMALE,MD,adm-clerical,polo,wife,0,0,06-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,VA,Riverwood,9760 4th Hwy,4,4,NO,2,1,?,77100,15420,7710,53970,Volkswagen,Jetta,1996,N
463,57,407958,20-07-1991,IL,250/500,500,1063.67,0,463356,MALE,Masters,priv-house-serv,dancing,wife,0,0,03-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Springfield,2509 Rock Drive,14,1,NO,2,0,YES,59400,6600,6600,46200,Dodge,Neon,1995,N
303,46,832300,14-01-2005,IN,100/300,1000,709.14,0,450184,MALE,Masters,machine-op-inspct,kayaking,husband,0,0,12-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Northbrook,2063 Weaver St,3,3,NO,1,0,NO,54890,9980,4990,39920,Toyota,Corolla,2006,N
137,30,927205,16-12-2011,IL,250/500,500,1039.55,0,466393,MALE,MD,exec-managerial,kayaking,unmarried,55600,-59700,16-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,WV,Columbus,3818 Texas Ridge,4,3,?,2,1,?,74030,6730,13460,53840,Chevrolet,Tahoe,2005,N
56,42,655356,07-07-1996,IL,250/500,500,1339.39,0,471786,FEMALE,Associate,adm-clerical,chess,not-in-family,0,0,25-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,NY,Columbus,3929 Elm Ave,13,1,?,1,2,YES,61490,11180,11180,39130,BMW,X5,1998,N
75,27,831053,05-08-1992,IN,250/500,1000,1202.75,0,602289,MALE,High School,handlers-cleaners,board-games,not-in-family,57900,-90100,21-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,SC,Northbrook,9911 Britain Lane,23,3,NO,0,2,YES,79560,6630,13260,59670,Volkswagen,Passat,2003,N
131,33,432740,09-10-1990,IL,100/300,2000,1081.17,0,445120,MALE,MD,sales,yachting,wife,0,-65200,28-01-2015,Parked Car,?,Minor Damage,Police,NY,Northbend,3246 Britain Ridge,3,1,?,0,1,NO,4900,490,490,3920,Toyota,Camry,2010,N
153,34,893853,27-02-1994,IL,250/500,500,991.39,0,449260,MALE,High School,tech-support,paintball,other-relative,45600,-61400,27-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Springfield,2696 Cherokee Ridge,0,3,?,1,0,NO,77770,14140,14140,49490,Nissan,Pathfinder,2000,N
255,43,594988,06-05-2007,IN,500/1000,500,984.02,0,472724,FEMALE,JD,transport-moving,board-games,unmarried,75800,0,10-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,SC,Columbus,5249 4th Ave,14,2,?,2,2,?,74700,7470,14940,52290,Nissan,Maxima,2007,Y
103,26,979544,21-04-2014,IL,100/300,500,1354.83,0,475173,MALE,MD,tech-support,sleeping,husband,66300,0,18-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NY,Arlington,4721 Cherokee Hwy,14,2,NO,2,2,?,40600,4060,4060,32480,Volkswagen,Passat,2010,N
97,28,191891,11-02-2010,OH,100/300,1000,830.31,0,443854,MALE,JD,farming-fishing,video-games,husband,0,-32600,13-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Springfield,8212 Flute Ridge,22,1,?,1,0,?,45270,10060,10060,25150,Jeep,Wrangler,2006,N
214,36,831479,04-06-2000,IL,100/300,2000,987.42,7000000,461418,FEMALE,Associate,machine-op-inspct,kayaking,own-child,0,0,31-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Springfield,3592 MLK Ridge,17,3,NO,0,3,?,47080,4280,8560,34240,Saab,93,2006,N
438,57,714346,05-10-1991,OH,500/1000,500,1119.29,0,616164,FEMALE,MD,machine-op-inspct,chess,husband,0,0,15-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,SC,Hillsdale,6494 4th Ave,14,1,?,0,0,NO,40700,4070,4070,32560,Volkswagen,Passat,2000,Y
87,27,326289,03-01-2004,OH,100/300,500,1048.39,0,620962,FEMALE,Masters,transport-moving,polo,own-child,0,0,13-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,NY,Riverwood,6608 Apache Lane,2,1,?,2,1,YES,34650,6300,3150,25200,Ford,F150,1996,N
27,28,944537,23-07-1992,OH,500/1000,1000,1074.47,0,465201,MALE,MD,protective-serv,camping,wife,0,0,13-02-2015,Parked Car,?,Trivial Damage,Police,WV,Riverwood,1553 Lincoln St,4,1,?,2,0,?,3200,400,400,2400,Jeep,Grand Cherokee,2012,N
206,42,779156,10-10-1993,IL,500/1000,1000,1230.76,0,470488,MALE,High School,machine-op-inspct,yachting,unmarried,0,-74200,03-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,VA,Hillsdale,7628 4th Lane,2,1,?,1,1,NO,78980,7180,14360,57440,Saab,92x,1997,Y
127,31,856153,09-07-2002,OH,500/1000,500,1255.02,0,462250,MALE,Associate,sales,reading,not-in-family,58200,0,13-01-2015,Parked Car,?,Minor Damage,None,SC,Riverwood,3028 5th St,10,1,YES,1,0,NO,6160,560,1120,4480,Nissan,Ultima,1996,N
422,60,473338,14-11-2010,IN,100/300,1000,1555.52,0,436408,FEMALE,MD,machine-op-inspct,hiking,own-child,43600,-67800,14-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,NC,Northbrook,8949 Rock Hwy,11,1,YES,1,1,NO,85250,15500,7750,62000,Mercedes,E400,1999,N
303,50,521694,03-03-1997,IL,100/300,2000,836.11,5000000,464230,MALE,Masters,sales,camping,not-in-family,0,0,20-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,SC,Springfield,9751 Tree St,1,4,YES,1,2,NO,72840,12140,6070,54630,Dodge,Neon,2010,N
228,40,136520,01-03-1997,IN,100/300,500,1450.98,0,478609,MALE,Associate,exec-managerial,base-jumping,husband,43700,0,18-01-2015,Parked Car,?,Trivial Damage,Police,WV,Northbend,4702 Texas Drive,20,1,?,0,2,NO,6050,1100,1100,3850,Toyota,Camry,2010,N
239,39,730819,18-08-1990,IN,250/500,2000,625.08,0,437156,FEMALE,JD,protective-serv,hiking,wife,44200,-37000,03-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Riverwood,2757 4th Hwy,10,1,NO,2,3,YES,87890,15980,7990,63920,BMW,X6,2014,Y
330,47,912665,28-05-2014,IL,100/300,2000,1133.27,0,432218,FEMALE,High School,craft-repair,chess,wife,0,-56400,01-03-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,NY,Northbrook,6678 Weaver Drive,20,2,?,0,2,YES,60500,11000,5500,44000,Ford,F150,1999,Y
128,35,469966,22-07-2004,IN,500/1000,500,1366.6,0,620493,FEMALE,MD,machine-op-inspct,kayaking,other-relative,0,0,01-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,NY,Columbus,8667 Weaver Lane,10,3,?,2,1,NO,88220,16040,16040,56140,Chevrolet,Malibu,2008,N
147,37,952300,02-08-2009,OH,500/1000,1000,1439.9,6000000,475391,FEMALE,Associate,prof-specialty,video-games,husband,0,-48400,29-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,SC,Springfield,4931 Maple Drive,2,2,NO,1,2,NO,53680,9760,4880,39040,Accura,MDX,2004,N
287,45,322609,05-07-2007,OH,500/1000,1000,1230.69,0,440720,MALE,Masters,transport-moving,golf,not-in-family,0,-54600,10-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,VA,Springfield,3808 5th Ave,19,1,NO,0,2,NO,53800,5380,5380,43040,Accura,MDX,2006,N
142,29,890280,24-01-2010,OH,100/300,2000,1307.68,0,606942,FEMALE,MD,craft-repair,dancing,husband,0,-48500,29-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,NY,Columbus,1725 Solo Lane,10,1,YES,1,2,NO,54360,4530,9060,40770,Mercedes,E400,1995,Y
162,35,431583,15-05-2000,IL,500/1000,2000,1124.69,0,446971,FEMALE,Masters,handlers-cleaners,bungie-jumping,husband,0,0,20-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,WV,Arlington,8097 Maple Lane,14,1,YES,0,3,YES,54340,9880,4940,39520,Toyota,Camry,2001,N
140,35,155912,21-03-2008,OH,100/300,1000,1520.78,0,470538,FEMALE,High School,craft-repair,chess,wife,0,-42900,21-01-2015,Parked Car,?,Trivial Damage,None,SC,Columbus,3320 5th Hwy,5,1,?,0,2,YES,2860,520,260,2080,Chevrolet,Tahoe,1997,Y
106,28,110143,07-05-1990,OH,100/300,2000,1609.11,0,601177,MALE,High School,craft-repair,polo,own-child,0,0,18-01-2015,Vehicle Theft,?,Minor Damage,Police,WV,Hillsdale,9573 2nd Ave,8,1,YES,2,1,YES,5490,0,1220,4270,Saab,95,1999,N
292,45,808544,05-02-1991,IL,500/1000,1000,1358.91,0,451470,MALE,Masters,craft-repair,dancing,unmarried,0,0,09-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Northbend,8336 1st Ridge,4,1,NO,0,2,?,7370,670,1340,5360,Suburu,Impreza,1997,N
34,34,409074,19-03-1992,OH,500/1000,500,1295.87,0,438529,FEMALE,PhD,priv-house-serv,chess,husband,0,0,13-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NC,Columbus,3998 4th Hwy,4,3,?,1,0,NO,50800,5080,5080,40640,Audi,A3,1997,Y
290,48,824728,24-04-2013,IL,250/500,500,1161.03,5000000,469742,MALE,Associate,adm-clerical,video-games,not-in-family,45300,0,13-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,VA,Northbrook,3966 Oak Hwy,20,2,?,1,1,?,41520,5190,5190,31140,Nissan,Ultima,2014,N
182,38,606037,10-04-2009,OH,500/1000,2000,1441.06,0,435534,FEMALE,Masters,armed-forces,movies,husband,53800,-78300,08-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,NY,Riverwood,7601 Andromedia Lane,18,1,?,2,3,YES,89650,8150,16300,65200,Dodge,RAM,2005,N
362,55,636843,01-12-2008,OH,100/300,1000,1097.99,0,442239,FEMALE,JD,other-service,base-jumping,unmarried,44400,-71500,17-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,NY,Hillsdale,5160 2nd Hwy,0,3,NO,0,3,NO,39690,0,0,39690,Suburu,Legacy,1998,N
143,32,111874,05-07-2000,IL,500/1000,1000,1464.42,0,468986,FEMALE,High School,exec-managerial,golf,husband,79900,0,15-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Springfield,3288 Tree Lane,1,1,?,2,0,NO,62260,5660,5660,50940,Saab,92x,1995,N
183,38,439844,11-06-2014,IL,250/500,500,1543.68,0,606988,FEMALE,Masters,prof-specialty,paintball,not-in-family,20200,0,24-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,NC,Hillsdale,5874 1st Hwy,17,1,NO,0,1,YES,51920,9440,4720,37760,Audi,A3,2001,Y
254,40,463513,23-04-1995,IL,250/500,500,1390.89,5000000,453719,MALE,College,armed-forces,bungie-jumping,wife,0,-74400,28-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,NY,Riverwood,6467 Best Ave,23,1,?,2,2,?,53460,9720,4860,38880,Volkswagen,Jetta,2009,N
249,43,577858,16-09-1990,OH,100/300,2000,1148.58,0,475524,FEMALE,MD,adm-clerical,golf,not-in-family,0,-71200,16-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Arlington,6309 5th Ave,4,3,YES,2,1,YES,57100,5710,5710,45680,Honda,CRV,2014,N
169,36,607351,11-12-1998,IN,250/500,500,1616.26,0,617804,MALE,High School,exec-managerial,yachting,unmarried,50700,-57600,09-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,NY,Springfield,8212 Rock Ave,0,3,NO,2,3,?,77440,14080,7040,56320,Dodge,Neon,2004,N
235,40,682754,09-10-1995,IL,500/1000,500,1398.94,0,613399,MALE,College,craft-repair,bungie-jumping,husband,0,0,24-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Riverwood,4107 MLK Ridge,11,1,NO,1,2,NO,68300,6830,13660,47810,Suburu,Forrestor,2003,N
112,32,757352,21-12-1999,OH,500/1000,1000,1238.92,0,453400,MALE,Associate,other-service,base-jumping,other-relative,57800,-53700,11-02-2015,Parked Car,?,Trivial Damage,None,NY,Arlington,4558 3rd Hwy,4,1,?,0,2,?,5060,460,920,3680,Honda,CRV,2012,N
16,32,307469,28-07-2002,IL,100/300,1000,968.46,0,615767,MALE,MD,tech-support,chess,not-in-family,50800,-66200,27-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,WV,Columbus,1762 Maple Hwy,0,3,?,0,2,?,59400,6600,13200,39600,Dodge,RAM,1995,Y
128,31,526296,03-08-1993,IL,100/300,500,1045.12,0,615311,FEMALE,High School,transport-moving,cross-fit,other-relative,0,-28300,19-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Springfield,5532 Francis Lane,2,3,NO,0,2,?,69930,0,15540,54390,Ford,Escape,2013,Y
103,27,658816,16-12-2007,IN,100/300,1000,1537.33,0,468470,FEMALE,College,handlers-cleaners,board-games,husband,0,-74800,20-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NY,Springfield,6158 Sky Ridge,11,3,YES,1,1,NO,77700,7770,15540,54390,Jeep,Wrangler,2008,N
356,54,913337,10-02-2008,OH,500/1000,500,912.3,0,461383,MALE,College,prof-specialty,yachting,wife,58500,-44000,10-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,SC,Northbend,9214 Elm Ridge,23,3,NO,2,1,?,68750,12500,12500,43750,Audi,A5,2007,Y
109,29,488464,01-10-2006,OH,100/300,2000,1007.28,6000000,457727,FEMALE,High School,adm-clerical,movies,husband,0,0,21-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,NY,Riverwood,1833 Solo Ave,17,3,NO,1,3,YES,91080,16560,16560,57960,Jeep,Wrangler,1995,N
2,20,480094,09-03-2003,IN,500/1000,1000,1189.98,4000000,613327,FEMALE,High School,craft-repair,golf,other-relative,0,-54700,01-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,WV,Columbus,1953 Sky Lane,22,1,NO,1,3,YES,48360,4030,8060,36270,Audi,A5,2000,N
198,34,263108,29-05-2003,OH,250/500,1000,1576.41,0,614941,MALE,Associate,handlers-cleaners,kayaking,other-relative,0,-55100,25-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,WV,Hillsdale,6834 1st Drive,18,1,YES,1,1,YES,95000,9500,9500,76000,Ford,F150,2001,N
107,32,298412,06-05-2002,OH,100/300,500,1172.82,4000000,440680,MALE,Associate,machine-op-inspct,yachting,other-relative,82100,0,24-02-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Arlington,9562 4th Ridge,8,1,?,1,3,NO,3900,780,390,2730,Ford,F150,2010,N
252,39,261905,28-02-2004,IL,500/1000,500,1312.22,9000000,609949,MALE,High School,transport-moving,cross-fit,other-relative,0,-33300,21-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,NC,Riverwood,4835 Britain Ridge,15,3,?,0,3,NO,59400,11880,5940,41580,Jeep,Grand Cherokee,2010,Y
303,43,674485,14-01-1999,OH,500/1000,1000,671.01,7000000,479655,FEMALE,Associate,machine-op-inspct,camping,other-relative,42900,-61500,08-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,NC,Springfield,8548 Cherokee Ridge,20,1,?,2,0,?,60210,6690,6690,46830,Nissan,Maxima,2013,N
101,32,223404,23-01-2002,IL,250/500,500,895.14,0,439964,MALE,JD,sales,video-games,other-relative,52600,-30400,10-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,SC,Columbus,2352 MLK Drive,4,1,?,0,3,YES,43600,8720,4360,30520,Suburu,Legacy,2010,N
446,57,991480,09-12-1992,IN,100/300,2000,1373.21,0,478486,MALE,College,adm-clerical,sleeping,unmarried,42700,-64900,14-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,SC,Northbrook,9734 2nd Ridge,10,3,NO,0,0,NO,62800,6280,12560,43960,Jeep,Wrangler,2012,N
330,48,804219,24-06-1998,OH,250/500,1000,1625.65,0,466498,MALE,College,farming-fishing,skydiving,husband,0,0,26-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,VA,Springfield,3122 Apache Drive,10,1,YES,1,3,?,59500,11900,5950,41650,Dodge,Neon,2006,N
211,37,483088,06-01-2011,OH,250/500,2000,1295.63,4000000,430878,FEMALE,PhD,armed-forces,skydiving,not-in-family,42200,-33800,30-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Northbend,9816 Britain St,22,3,YES,1,0,?,53460,5940,5940,41580,Honda,CRV,2009,N
172,33,100804,24-02-2012,IL,100/300,1000,1459.96,6000000,600127,FEMALE,High School,adm-clerical,reading,wife,41300,-42000,07-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,NY,Northbend,8214 Flute St,15,3,NO,0,1,NO,41690,7580,7580,26530,Saab,95,1999,N
316,46,941807,25-06-2011,OH,100/300,500,1219.94,7000000,431968,FEMALE,Masters,prof-specialty,paintball,wife,0,-51000,27-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Arlington,6259 Lincoln Hwy,13,1,?,0,1,YES,63100,6310,12620,44170,Accura,TL,2000,N
435,60,593466,21-11-2006,OH,500/1000,500,1064.49,5000000,462804,MALE,Associate,priv-house-serv,chess,other-relative,73500,-43300,13-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,WV,Arlington,4492 Andromedia Ave,23,1,NO,2,1,?,62880,5240,10480,47160,Mercedes,E400,2007,Y
344,51,437442,27-06-2008,IL,100/300,1000,959.83,0,435809,FEMALE,Masters,sales,paintball,not-in-family,0,-38700,02-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,NY,Northbend,6179 3rd Ridge,18,3,?,2,2,YES,75400,17400,11600,46400,BMW,X6,2006,Y
204,40,942106,30-08-1993,OH,250/500,2000,1767.02,0,453193,MALE,JD,machine-op-inspct,hiking,husband,0,-49300,27-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Arlington,3799 Embaracadero Drive,7,1,YES,1,1,NO,46200,4200,8400,33600,Audi,A5,1997,N
278,47,794951,21-04-2008,IN,500/1000,500,1285.01,0,459630,MALE,Masters,machine-op-inspct,sleeping,not-in-family,0,-39800,02-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,VA,Hillsdale,5071 1st Lane,21,1,NO,2,2,YES,58500,5850,5850,46800,Toyota,Camry,2010,N
434,57,182450,23-06-2000,OH,500/1000,2000,1422.95,0,608982,MALE,JD,transport-moving,bungie-jumping,husband,0,0,17-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Columbus,6574 4th Drive,15,3,NO,1,3,NO,66240,11040,11040,44160,Nissan,Maxima,2003,N
209,36,730973,11-01-2010,IN,100/300,2000,1223.39,0,452218,FEMALE,MD,craft-repair,camping,wife,0,0,12-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,PA,Hillsdale,2711 Britain Ave,17,3,?,1,3,?,65440,8180,8180,49080,Jeep,Wrangler,2014,N
250,43,687755,28-03-1990,IL,500/1000,2000,1539.06,0,434150,MALE,Masters,sales,exercise,other-relative,37800,0,20-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,SC,Hillsdale,4214 MLK Ridge,2,1,NO,0,3,?,64200,10700,10700,42800,Ford,F150,2002,N
61,25,757644,29-01-1998,IN,100/300,2000,988.06,0,460579,FEMALE,Masters,other-service,dancing,not-in-family,0,0,05-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Northbend,7976 Britain Drive,1,3,YES,1,0,NO,32320,4040,4040,24240,Dodge,RAM,2000,N
80,28,998865,05-12-2014,IL,500/1000,1000,1740.57,0,442142,FEMALE,College,farming-fishing,golf,wife,0,-18600,20-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,SC,Northbend,4995 Weaver Ridge,3,1,?,0,1,?,33480,3720,3720,26040,Dodge,Neon,2011,N
25,38,944953,07-12-1995,OH,250/500,1000,1540.91,7000000,608807,MALE,College,adm-clerical,polo,wife,40900,0,24-02-2015,Vehicle Theft,?,Trivial Damage,None,VA,Northbend,1515 Embaracadero St,0,1,NO,1,1,YES,4320,480,960,2880,Toyota,Corolla,1995,N
4,29,386429,27-05-2002,IL,250/500,500,1381.88,5000000,433153,MALE,High School,tech-support,exercise,other-relative,0,-77800,21-02-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Hillsdale,2968 Andromedia Ave,4,1,NO,0,2,?,4200,840,420,2940,Jeep,Wrangler,2008,N
32,29,108270,09-08-2002,OH,100/300,500,1446.98,0,436560,MALE,MD,adm-clerical,sleeping,own-child,0,-45700,11-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Arlington,9236 2nd Hwy,11,3,YES,2,0,NO,57970,10540,5270,42160,Saab,93,2006,N
125,31,205134,02-12-2012,IN,500/1000,500,1220.86,0,436784,MALE,JD,other-service,paintball,husband,55400,-40400,24-01-2015,Parked Car,?,Trivial Damage,None,NY,Arlington,9639 Britain Ridge,4,1,YES,1,2,?,4320,0,960,3360,Saab,93,2003,N
276,45,749325,22-03-2000,IL,500/1000,500,948.1,0,430621,FEMALE,High School,machine-op-inspct,reading,wife,44500,-61400,06-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,SC,Columbus,9422 Washington Ridge,11,3,?,0,2,?,69300,13860,6930,48510,Ford,Escape,2010,N
148,30,774303,13-04-2002,OH,100/300,500,1471.24,0,601574,FEMALE,Masters,farming-fishing,camping,own-child,57500,-93600,15-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,NC,Riverwood,1213 4th Lane,4,3,NO,2,2,NO,32480,4060,4060,24360,Dodge,Neon,1997,N
222,38,698470,17-06-2008,IN,100/300,2000,1157.97,0,433853,MALE,MD,machine-op-inspct,hiking,not-in-family,0,-64500,22-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,NY,Hillsdale,3872 5th Drive,20,1,?,2,2,YES,60480,6720,6720,47040,Accura,TL,2001,N
32,38,719989,07-04-1994,IL,250/500,2000,566.11,5000000,453164,MALE,Associate,armed-forces,polo,unmarried,0,0,21-01-2015,Parked Car,?,Trivial Damage,Police,NC,Springfield,9397 5th Hwy,22,1,YES,0,0,NO,2640,440,440,1760,Honda,CRV,2015,N
78,27,309323,29-02-1992,OH,500/1000,500,1060.88,0,613931,MALE,JD,other-service,skydiving,other-relative,0,-66500,03-02-2015,Parked Car,?,Trivial Damage,Police,SC,Northbend,8876 1st St,4,1,NO,0,3,NO,6050,550,1100,4400,Volkswagen,Passat,2009,N
238,43,444035,11-05-1996,OH,250/500,1000,1524.45,4000000,607458,MALE,High School,handlers-cleaners,chess,wife,0,-44800,16-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,NC,Hillsdale,3397 5th Ave,21,4,NO,0,0,NO,42700,4270,4270,34160,Saab,92x,1995,Y
313,47,431478,03-04-2013,IN,250/500,1000,1556.17,0,463835,MALE,College,prof-specialty,reading,wife,63900,-53300,07-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Ambulance,SC,Hillsdale,3263 Pine Ridge,20,1,YES,1,3,NO,40260,3660,7320,29280,Accura,MDX,1996,Y
334,50,797634,12-11-2009,OH,500/1000,500,1216.24,0,613945,MALE,Masters,priv-house-serv,polo,wife,26700,-47200,14-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NY,Columbus,8639 5th Hwy,7,1,NO,2,0,?,50000,5000,10000,35000,Chevrolet,Silverado,2008,N
190,35,284836,05-11-2008,IN,250/500,500,1484.72,5000000,432699,FEMALE,High School,tech-support,golf,husband,0,0,02-02-2015,Parked Car,?,Trivial Damage,None,NY,Riverwood,5743 4th Ridge,4,1,NO,0,1,NO,3840,640,320,2880,Saab,92x,1998,N
194,41,238196,15-02-1993,IL,250/500,500,1203.81,0,613119,MALE,JD,transport-moving,video-games,not-in-family,52500,-51300,06-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,WV,Springfield,3555 Francis Ridge,17,3,?,0,2,?,95900,13700,20550,61650,Saab,95,1999,N
290,47,885789,21-07-2008,IN,250/500,1000,1393.34,0,472922,MALE,High School,exec-managerial,bungie-jumping,other-relative,0,-61400,15-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Northbend,4939 Oak Lane,20,3,YES,1,3,YES,56160,6240,12480,37440,Audi,A5,2002,N
26,42,287436,11-09-2010,OH,100/300,1000,1484.48,0,613849,MALE,PhD,armed-forces,sleeping,not-in-family,50700,-36300,24-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,SC,Riverwood,3100 Best St,10,1,NO,2,3,?,63030,5730,11460,45840,Saab,92x,1996,N
254,41,496067,22-12-1995,IL,250/500,500,1581.27,5000000,603827,FEMALE,PhD,handlers-cleaners,skydiving,own-child,42200,-48000,07-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Riverwood,3029 5th Ave,8,1,YES,2,2,NO,63470,5770,11540,46160,BMW,X6,1999,N
199,38,206004,26-09-1991,IL,250/500,1000,1281.25,0,467780,FEMALE,High School,tech-support,movies,other-relative,0,-53100,04-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,WV,Columbus,8941 Solo Ridge,6,1,?,0,2,NO,44440,8080,4040,32320,BMW,X6,2007,Y
137,35,153027,11-03-2010,IN,250/500,500,1667.83,0,460586,MALE,JD,prof-specialty,paintball,husband,48500,-67400,04-02-2015,Parked Car,?,Minor Damage,Police,WV,Northbrook,4447 Francis Hwy,4,1,YES,1,1,NO,6600,1200,1200,4200,Jeep,Grand Cherokee,2005,N
134,36,469426,15-07-1990,OH,250/500,1000,1497.41,0,613842,MALE,PhD,machine-op-inspct,kayaking,husband,14100,-44500,25-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Northbrook,7701 Tree St,17,3,NO,2,0,YES,77200,9650,9650,57900,Ford,Escape,1996,N
73,30,654974,10-05-2009,OH,100/300,500,803.36,0,435371,FEMALE,High School,protective-serv,reading,husband,0,0,25-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,WV,Springfield,4653 Pine St,15,3,?,0,0,YES,57000,0,11400,45600,Audi,A3,2013,N
289,45,943425,28-10-1999,OH,250/500,2000,1221.41,0,466289,FEMALE,Masters,farming-fishing,movies,wife,46700,-72500,27-02-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Riverwood,8742 4th St,9,1,NO,2,1,NO,2700,300,300,2100,Honda,Accord,2006,N
176,36,641845,30-03-1995,OH,250/500,500,1865.83,5000000,436173,MALE,College,transport-moving,kayaking,unmarried,32800,-50600,11-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,WV,Hillsdale,7316 Texas Ave,17,1,?,2,1,YES,47300,4300,8600,34400,Volkswagen,Jetta,2006,N
145,37,794534,14-12-1991,OH,250/500,2000,1434.27,0,457234,FEMALE,Associate,tech-support,sleeping,unmarried,0,-35900,04-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,VA,Arlington,2950 MLK Ave,13,1,?,2,3,?,55110,5010,10020,40080,Nissan,Maxima,2002,N
164,31,357808,31-01-2011,IN,500/1000,500,1114.68,0,474758,FEMALE,Associate,other-service,reading,husband,44500,-55900,26-01-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Springfield,8233 Tree Drive,5,1,YES,1,0,NO,4320,480,480,3360,Mercedes,E400,2002,N
186,38,536052,21-04-2006,OH,250/500,2000,1218.56,0,477373,FEMALE,Masters,transport-moving,video-games,husband,39300,-60300,01-03-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,VA,Arlington,4642 Rock Ridge,23,3,YES,2,2,?,68760,11460,5730,51570,Saab,95,1998,Y
85,31,873384,10-03-2004,IL,250/500,2000,1234.69,9000000,613471,FEMALE,MD,tech-support,paintball,husband,0,0,06-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,WV,Arlington,7733 Britain Lane,1,2,NO,2,1,?,74400,14880,7440,52080,BMW,M5,2003,Y
162,33,790225,05-01-1991,OH,250/500,500,964.92,0,601581,FEMALE,Associate,exec-managerial,base-jumping,other-relative,45700,0,09-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NY,Hillsdale,2753 Cherokee Ave,17,4,NO,2,1,NO,35300,3530,3530,28240,Mercedes,E400,1996,Y
396,57,587498,15-10-1996,IL,500/1000,500,1351.72,0,612102,MALE,High School,tech-support,camping,wife,0,-49400,05-02-2015,Parked Car,?,Minor Damage,None,NY,Springfield,3995 Lincoln Hwy,3,1,YES,1,3,?,2640,480,480,1680,Volkswagen,Passat,2000,N
270,41,639027,21-06-1994,IL,250/500,1000,817.28,0,460263,MALE,High School,sales,cross-fit,unmarried,62200,0,03-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Columbus,4095 MLK St,17,3,?,1,1,NO,60190,4630,9260,46300,Mercedes,ML350,2014,Y
168,39,217899,13-06-1994,IL,500/1000,1000,1389.59,0,479134,FEMALE,Masters,machine-op-inspct,exercise,own-child,0,-42600,24-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NY,Northbend,5782 Rock Drive,23,3,YES,1,3,NO,41580,3780,7560,30240,Chevrolet,Malibu,2015,Y
274,45,589094,27-05-2003,IN,250/500,1000,1353.53,0,451467,FEMALE,JD,tech-support,cross-fit,unmarried,54700,-47900,14-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Columbus,2900 Sky Drive,13,1,YES,0,0,NO,58500,11700,0,46800,Accura,MDX,1995,Y
263,43,458829,06-07-1996,IN,500/1000,1000,1294.04,0,602670,FEMALE,Masters,handlers-cleaners,movies,not-in-family,0,0,08-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Riverwood,1515 Pine Lane,17,1,YES,2,3,YES,79320,13220,6610,59490,Nissan,Ultima,1997,N
152,33,626208,08-05-2005,OH,100/300,1000,840.81,0,613607,FEMALE,High School,farming-fishing,chess,husband,0,0,14-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Arlington,4876 Washington Drive,2,1,YES,0,2,?,82610,7510,7510,67590,Ford,Escape,2002,Y
46,41,315041,02-11-2010,OH,100/300,2000,998.19,0,611556,FEMALE,MD,priv-house-serv,video-games,husband,43700,-66300,23-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Hillsdale,5779 2nd Lane,23,3,NO,1,3,?,78600,13100,19650,45850,Dodge,RAM,2004,Y
276,46,283267,29-07-2012,OH,100/300,2000,1090.32,0,435518,MALE,College,handlers-cleaners,chess,husband,0,-70400,05-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,SC,Columbus,6706 Francis Drive,17,1,NO,1,2,?,51390,5710,5710,39970,Volkswagen,Jetta,2007,Y
234,44,442494,06-06-2002,IN,500/1000,500,1780.67,0,465942,MALE,Associate,other-service,exercise,other-relative,0,0,19-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,NC,Springfield,6384 5th Ridge,3,1,NO,1,0,NO,70200,7020,7020,56160,Ford,F150,2012,Y
64,30,159243,19-09-1991,IL,250/500,2000,1681.01,0,446174,MALE,JD,protective-serv,base-jumping,own-child,0,-51100,07-02-2015,Parked Car,?,Minor Damage,None,SC,Riverwood,3006 Lincoln Ridge,16,1,NO,2,1,NO,4900,490,1470,2940,Jeep,Wrangler,2015,N
456,62,669800,24-06-2009,OH,250/500,1000,1395.77,0,611651,FEMALE,MD,protective-serv,chess,own-child,82600,-49500,07-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,PA,Hillsdale,5352 Lincoln Drive,13,3,?,1,3,NO,66480,5540,11080,49860,Saab,92x,2012,Y
58,23,520179,29-05-1992,OH,500/1000,2000,1471.44,5000000,446657,MALE,High School,transport-moving,reading,own-child,57500,0,20-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NC,Riverwood,6110 Rock Ridge,8,1,NO,2,1,NO,50380,4580,9160,36640,Chevrolet,Tahoe,2007,Y
475,61,607974,12-08-2004,IL,500/1000,500,1265.72,0,612506,FEMALE,Masters,handlers-cleaners,paintball,wife,0,-59500,18-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,SC,Columbus,7797 Tree Ridge,23,1,YES,0,2,?,64350,9900,9900,44550,Mercedes,E400,1998,N
96,29,465065,24-12-2006,IN,250/500,1000,1274.7,5000000,618493,MALE,College,prof-specialty,hiking,other-relative,47500,-58700,11-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,NC,Springfield,4910 1st Lane,15,3,YES,2,3,YES,55400,5540,11080,38780,Chevrolet,Silverado,2004,Y
99,28,369941,24-07-2007,OH,100/300,500,1330.39,0,612664,MALE,MD,prof-specialty,basketball,wife,0,0,22-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NY,Columbus,8766 Lincoln Lane,3,1,NO,2,2,YES,49900,4990,9980,34930,Dodge,Neon,1998,N
38,28,447226,17-08-1994,OH,500/1000,500,1122.95,4000000,473653,MALE,Masters,priv-house-serv,golf,other-relative,78000,0,23-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,WV,Northbrook,6399 Oak Drive,4,3,NO,0,3,YES,74880,12480,12480,49920,Accura,TL,2000,N
259,44,831668,10-04-1996,OH,250/500,2000,1655.79,0,454529,FEMALE,Masters,exec-managerial,exercise,husband,0,0,15-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,NY,Northbend,3127 Flute St,8,1,YES,0,0,YES,105820,16280,16280,73260,Audi,A3,2002,N
241,43,922937,11-12-1992,IN,250/500,1000,935.77,0,437422,MALE,Associate,prof-specialty,golf,own-child,0,-36000,20-02-2015,Vehicle Theft,?,Trivial Damage,Police,PA,Northbend,8920 Best Ave,21,1,NO,1,0,YES,7150,1300,650,5200,Volkswagen,Jetta,2003,N
437,58,640474,01-08-2010,IN,500/1000,2000,1192.04,0,619470,MALE,Associate,craft-repair,dancing,own-child,66100,-31400,19-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Northbrook,7314 Tree Drive,23,1,YES,0,0,NO,55800,11160,11160,33480,Dodge,RAM,2004,N
130,34,153298,23-03-2009,OH,100/300,500,990.11,0,442666,MALE,Masters,sales,kayaking,other-relative,0,-41200,10-01-2015,Parked Car,?,Trivial Damage,None,NY,Riverwood,8872 Oak Ridge,8,1,?,1,3,YES,5830,1060,1060,3710,Dodge,RAM,2015,N
269,41,334749,29-07-1996,OH,100/300,2000,1422.21,0,620507,FEMALE,Associate,handlers-cleaners,polo,unmarried,0,-46400,16-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,WV,Riverwood,5022 1st St,21,1,YES,2,1,NO,85900,17180,17180,51540,Suburu,Forrestor,2005,Y
103,29,221283,23-08-1994,OH,500/1000,500,914.85,0,614867,MALE,Associate,prof-specialty,base-jumping,other-relative,72100,0,12-02-2015,Parked Car,?,Minor Damage,Police,OH,Columbus,3423 Francis Ave,5,1,NO,2,3,NO,7110,790,1580,4740,Accura,MDX,2005,N
284,43,961496,05-01-1992,IL,250/500,500,1123.84,0,609898,MALE,PhD,prof-specialty,kayaking,other-relative,48200,0,23-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Columbus,9529 4th Drive,12,3,NO,0,0,YES,36960,6720,3360,26880,Chevrolet,Tahoe,2007,N
189,39,804751,11-09-1997,OH,250/500,2000,838.02,0,450702,FEMALE,College,tech-support,movies,own-child,0,0,13-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,VA,Northbrook,1818 Tree St,7,3,?,2,0,YES,64400,6440,6440,51520,Dodge,Neon,1997,N
267,43,369226,10-02-2002,OH,250/500,500,1300.68,0,600418,MALE,PhD,adm-clerical,sleeping,unmarried,49000,0,27-01-2015,Parked Car,?,Minor Damage,None,NC,Northbend,4431 Rock St,0,1,NO,1,3,YES,1920,480,0,1440,Chevrolet,Tahoe,2011,N
39,22,691115,28-01-1993,IN,500/1000,500,1173.21,0,431202,MALE,JD,farming-fishing,polo,not-in-family,0,0,14-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,SC,Northbend,4782 Sky Lane,14,1,YES,0,1,YES,86130,15660,7830,62640,Suburu,Legacy,2009,Y
140,32,713172,23-10-1996,IL,250/500,1000,985.97,5000000,457793,FEMALE,College,protective-serv,cross-fit,other-relative,0,0,01-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,VA,Northbrook,7112 Weaver Ave,13,1,?,2,3,?,82170,14940,7470,59760,Chevrolet,Silverado,1995,Y
243,41,621756,21-04-1997,IN,100/300,1000,1129.23,0,470190,FEMALE,College,farming-fishing,camping,own-child,17300,-60400,23-02-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,WV,Hillsdale,4020 Best Drive,1,1,?,1,0,YES,50300,10060,5030,35210,Suburu,Legacy,1999,Y
116,31,615116,09-11-2008,IN,250/500,500,1194.83,0,603733,FEMALE,MD,prof-specialty,camping,husband,28600,0,20-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,SC,Riverwood,2037 5th Drive,23,1,NO,0,0,NO,44200,4420,8840,30940,Suburu,Forrestor,1997,N
219,43,947598,20-06-2002,IN,100/300,1000,1114.29,0,465136,FEMALE,High School,transport-moving,polo,other-relative,51300,0,08-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,VA,Northbrook,4699 Texas Ridge,1,1,?,2,2,YES,66660,6060,6060,54540,Toyota,Highlander,2006,N
96,26,658002,21-10-2005,OH,250/500,2000,1509.04,0,611723,FEMALE,Associate,prof-specialty,bungie-jumping,husband,10000,0,23-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,SC,Riverwood,1832 Elm Hwy,9,1,YES,2,3,NO,78320,7120,14240,56960,Saab,92x,2007,N
149,34,374545,28-08-2005,IN,250/500,500,664.86,0,608963,FEMALE,PhD,craft-repair,skydiving,wife,0,-60000,04-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,WV,Columbus,5226 Maple St,3,1,?,0,1,NO,105040,16160,16160,72720,Dodge,RAM,1999,N
246,43,805806,16-01-2013,IN,250/500,1000,1267.4,6000000,454139,MALE,JD,adm-clerical,basketball,husband,0,0,09-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,NY,Hillsdale,3771 4th St,0,1,NO,2,1,?,50700,5070,5070,40560,Accura,RSX,2006,N
293,45,235097,28-04-1992,IL,100/300,1000,1119.23,0,447560,FEMALE,MD,exec-managerial,exercise,unmarried,51500,0,18-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Fire,WV,Northbend,8701 5th Lane,13,3,NO,1,1,NO,51210,11380,5690,34140,Jeep,Wrangler,2015,N
339,48,290971,10-10-2005,OH,100/300,500,1698.51,0,444378,MALE,JD,other-service,dancing,unmarried,0,0,10-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Hillsdale,7574 4th St,18,3,NO,2,1,?,51840,8640,8640,34560,Volkswagen,Jetta,2001,N
160,33,180286,08-02-2009,IL,500/1000,1000,1422.78,0,616583,FEMALE,High School,exec-managerial,exercise,husband,61600,0,20-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,NC,Riverwood,1989 Solo Lane,17,3,?,2,3,YES,52800,5280,5280,42240,Nissan,Pathfinder,2006,N
224,42,662088,06-03-2005,OH,500/1000,500,1212.75,0,455913,FEMALE,College,prof-specialty,kayaking,own-child,0,-51400,27-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Springfield,6331 MLK Ave,11,1,?,0,0,YES,55200,9200,13800,32200,Honda,Civic,1998,N
194,34,884365,17-05-1994,IN,100/300,1000,1423.34,0,454399,MALE,Associate,sales,camping,not-in-family,55300,-37900,21-01-2015,Vehicle Theft,?,Minor Damage,None,WV,Riverwood,8453 Elm St,0,1,YES,0,3,NO,9100,1400,1400,6300,Chevrolet,Malibu,2003,N
385,51,178081,20-07-1990,IN,250/500,1000,976.37,0,602842,FEMALE,MD,craft-repair,reading,husband,0,-61000,18-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,WV,Riverwood,1422 Flute Ave,14,3,?,1,3,?,67600,13520,6760,47320,Suburu,Legacy,2007,N
100,33,507452,17-04-2005,OH,250/500,500,1124.59,6000000,459428,MALE,College,adm-clerical,golf,not-in-family,67300,0,26-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,NC,Columbus,5058 4th Lane,4,1,NO,1,2,NO,40800,6800,6800,27200,BMW,X5,2004,N
371,50,990624,10-02-1994,IN,250/500,1000,1569.33,0,613114,MALE,PhD,machine-op-inspct,board-games,not-in-family,79600,0,29-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Other,WV,Springfield,3098 Oak Lane,2,3,NO,2,1,YES,84500,13000,13000,58500,Volkswagen,Passat,2011,Y
175,39,892148,29-03-1995,IN,500/1000,500,1359.36,5000000,450709,MALE,PhD,exec-managerial,hiking,husband,0,-43600,08-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,SC,Arlington,9103 MLK Lane,9,3,YES,2,2,YES,71610,13020,6510,52080,Toyota,Highlander,2012,Y
373,55,398683,30-04-2007,IN,250/500,500,1607.36,0,444626,MALE,MD,sales,yachting,own-child,0,0,19-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Arlington,8624 Francis Ave,21,4,?,0,2,NO,60600,6060,12120,42420,Dodge,RAM,2007,Y
258,41,605100,15-02-2001,IL,100/300,500,1042.25,0,601206,MALE,Masters,exec-managerial,reading,unmarried,0,-44400,08-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,NC,Riverwood,2905 Embaracadero Drive,0,3,NO,2,3,NO,81240,6770,20310,54160,Mercedes,C300,2008,Y
255,39,143109,09-07-2001,OH,250/500,500,1453.95,0,470389,FEMALE,PhD,armed-forces,bungie-jumping,not-in-family,38200,0,07-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Springfield,3443 Maple Ridge,17,3,YES,0,3,NO,29300,2930,5860,20510,Audi,A3,2010,N
37,31,230223,06-09-2008,IL,500/1000,500,1969.63,0,615218,FEMALE,MD,sales,skydiving,own-child,0,0,13-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Northbend,1618 Maple Hwy,21,3,NO,1,1,YES,76450,6950,13900,55600,Dodge,RAM,1995,N
322,44,769602,19-12-2004,IL,100/300,1000,1156.19,0,606249,FEMALE,College,machine-op-inspct,cross-fit,husband,49900,-62700,15-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,NY,Northbrook,3751 Tree Hwy,20,3,YES,0,3,?,49400,9880,4940,34580,Jeep,Wrangler,2010,N
204,38,420815,15-11-2000,IL,100/300,2000,1124.47,0,616161,FEMALE,MD,tech-support,kayaking,wife,0,-45100,14-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,SC,Northbrook,6848 Elm Hwy,5,1,NO,0,1,?,90530,16460,16460,57610,Ford,F150,2003,N
76,31,973546,14-03-2007,OH,500/1000,500,1493.5,5000000,442335,FEMALE,Associate,priv-house-serv,movies,not-in-family,39900,-44000,31-01-2015,Vehicle Theft,?,Minor Damage,Police,WV,Northbrook,4237 4th St,7,1,NO,2,1,NO,8030,1460,730,5840,Mercedes,E400,1995,N
193,40,608039,28-12-2004,IL,100/300,500,1155.38,0,604952,FEMALE,PhD,handlers-cleaners,movies,not-in-family,34200,-32300,28-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,NY,Columbus,6581 Rock Ridge,6,1,NO,0,0,YES,63900,6390,6390,51120,Accura,TL,2001,N
405,55,250162,05-07-1999,IL,250/500,500,878.19,0,441533,MALE,PhD,machine-op-inspct,golf,unmarried,57100,0,01-03-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,NC,Northbend,7236 Apache Lane,2,4,YES,0,2,NO,38640,4830,4830,28980,Chevrolet,Tahoe,1997,N
435,58,786432,15-11-1997,IN,100/300,2000,1145.85,0,471784,MALE,JD,sales,movies,not-in-family,0,-40000,10-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,NY,Columbus,3846 4th Hwy,19,3,?,1,1,YES,41490,9220,4610,27660,Mercedes,E400,2004,N
54,35,445195,27-09-2010,IN,100/300,500,1261.28,0,453265,FEMALE,MD,protective-serv,hiking,unmarried,68500,-42100,25-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,VA,Springfield,5028 Maple Ridge,21,3,YES,2,0,?,79090,14380,7190,57520,Nissan,Maxima,2012,N
144,35,938634,30-08-1993,IL,100/300,500,1427.46,0,444922,MALE,High School,machine-op-inspct,cross-fit,wife,0,0,30-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,WV,Springfield,7426 Rock Drive,3,3,?,0,0,?,87900,17580,8790,61530,Dodge,Neon,1995,N
92,32,482495,29-01-1998,IL,500/1000,500,1592.41,0,474324,MALE,Masters,prof-specialty,yachting,husband,58900,-29100,06-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Columbus,5771 Best St,22,1,?,2,3,YES,53400,5340,5340,42720,Jeep,Wrangler,1996,N
173,36,796005,18-08-2007,OH,250/500,1000,1274.63,0,441298,MALE,College,machine-op-inspct,basketball,unmarried,51000,0,08-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Fire,SC,Springfield,9818 Cherokee Ave,22,1,YES,2,3,NO,52030,9460,9460,33110,Accura,MDX,1995,N
436,60,910604,14-04-1992,IN,250/500,500,1362.31,0,446606,MALE,High School,prof-specialty,bungie-jumping,wife,67600,-65300,13-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,VA,Arlington,7819 2nd Ave,16,1,NO,0,2,NO,82060,14920,7460,59680,Saab,93,2005,N
155,35,327488,09-08-1993,OH,250/500,1000,919.37,0,459537,FEMALE,Associate,protective-serv,hiking,not-in-family,83600,0,18-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Northbrook,1331 Elm Ridge,0,1,?,0,3,?,48360,8060,8060,32240,Nissan,Maxima,1997,N
78,31,715202,02-04-1991,OH,250/500,1000,1377.23,0,440757,FEMALE,Masters,armed-forces,kayaking,unmarried,72600,0,01-03-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,NY,Hillsdale,9240 Britain Ave,1,3,?,2,1,?,52290,5810,11620,34860,Nissan,Maxima,1997,N
440,57,648852,15-03-2007,IL,100/300,1000,995.55,5000000,604948,MALE,College,protective-serv,paintball,wife,51500,-52100,02-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,SC,Arlington,6668 Andromedia Ridge,19,3,YES,0,3,?,68200,12400,12400,43400,Jeep,Wrangler,2007,Y
264,43,516959,01-05-2010,IL,100/300,500,1508.12,6000000,433275,MALE,PhD,craft-repair,basketball,wife,0,0,20-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,NY,Columbus,5276 2nd Lane,0,3,?,2,1,NO,60750,13500,6750,40500,Jeep,Wrangler,2015,Y
66,30,984456,24-06-2003,IN,500/1000,500,484.67,0,608309,FEMALE,College,adm-clerical,paintball,wife,21100,-60800,24-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,SC,Arlington,2889 Weaver St,2,3,?,0,2,YES,65560,11920,11920,41720,Volkswagen,Passat,2015,Y
366,50,801331,08-07-1990,IN,500/1000,1000,1561.41,0,462767,FEMALE,High School,handlers-cleaners,basketball,husband,21200,0,04-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Arlington,1879 4th Lane,5,3,YES,1,3,NO,70290,12780,12780,44730,Mercedes,C300,2012,N
188,37,786103,24-09-1994,OH,100/300,500,1457.21,0,471785,FEMALE,JD,adm-clerical,hiking,own-child,46300,0,17-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,SC,Columbus,5499 Elm Hwy,6,1,?,2,0,YES,45000,5000,5000,35000,Suburu,Forrestor,2003,N
224,39,684193,20-06-2012,IL,500/1000,1000,1128.71,0,601397,FEMALE,JD,prof-specialty,sleeping,other-relative,0,-47100,04-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,VA,Riverwood,8822 Sky St,21,3,YES,2,1,?,61800,12360,6180,43260,Suburu,Impreza,2007,N
253,46,247505,19-04-2006,IL,100/300,500,1358.2,0,477636,FEMALE,MD,transport-moving,movies,husband,52900,0,14-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Columbus,4254 Best Ridge,11,3,YES,0,0,NO,64570,5870,11740,46960,Jeep,Wrangler,2001,N
446,61,259792,07-04-1999,IL,100/300,1000,1232.79,0,441967,FEMALE,High School,adm-clerical,reading,unmarried,49900,-62100,07-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,VA,Springfield,5812 Weaver Ave,3,1,YES,2,0,YES,70500,7050,14100,49350,Suburu,Forrestor,2007,N
169,37,185124,07-12-2001,IL,100/300,1000,936.19,0,454776,MALE,JD,armed-forces,movies,other-relative,70600,-48500,02-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Northbrook,7155 Apache Drive,4,3,?,2,1,YES,57900,17370,5790,34740,Audi,A5,2005,N
255,46,760700,25-11-2006,IL,250/500,500,1302.34,0,431532,FEMALE,JD,prof-specialty,video-games,own-child,0,-52600,12-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbrook,1376 Pine St,2,3,NO,1,0,NO,57860,5260,10520,42080,Volkswagen,Jetta,2011,N
209,39,362407,06-12-1996,IN,100/300,500,1264.99,0,614169,MALE,PhD,transport-moving,polo,husband,67800,0,01-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,VA,Northbrook,3340 3rd Hwy,22,1,?,1,1,NO,37800,8400,4200,25200,Chevrolet,Silverado,1995,N
210,37,389525,10-07-2012,OH,500/1000,500,1467.76,0,601425,FEMALE,MD,tech-support,hiking,own-child,38700,-33100,22-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Ambulance,NY,Northbend,3097 4th Drive,8,1,?,1,3,YES,63300,6330,6330,50640,Toyota,Highlander,2000,N
174,33,179538,07-04-2014,IN,250/500,2000,1124.43,0,477346,FEMALE,College,farming-fishing,paintball,own-child,0,0,16-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,WV,Northbrook,1916 Elm St,14,3,YES,0,1,YES,44200,8840,4420,30940,Saab,93,1995,N
70,28,265437,11-10-2003,IL,250/500,1000,1319.81,0,613587,MALE,High School,machine-op-inspct,yachting,husband,67200,-59400,28-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,WV,Arlington,8917 Cherokee Lane,14,4,NO,1,0,YES,31680,3520,3520,24640,Toyota,Camry,2006,N
89,32,266247,17-01-2015,IN,100/300,2000,1482.53,0,620358,FEMALE,MD,tech-support,kayaking,not-in-family,49100,-45100,23-01-2015,Parked Car,?,Trivial Damage,Police,WV,Northbrook,8492 Weaver Hwy,5,1,YES,1,2,?,100,10,20,70,Audi,A3,2002,N
458,61,921851,07-12-1992,IN,100/300,1000,1328.18,0,617699,FEMALE,High School,protective-serv,bungie-jumping,other-relative,53000,0,25-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Columbus,3753 Francis Lane,18,1,NO,2,1,NO,56340,6260,6260,43820,Volkswagen,Passat,2003,N
239,40,488724,29-11-2004,IN,100/300,500,1463.95,0,430567,FEMALE,JD,sales,skydiving,own-child,0,0,11-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,NC,Springfield,4545 4th Ridge,20,3,?,0,0,YES,69740,6340,6340,57060,Dodge,Neon,2003,N
161,38,192524,02-01-2004,IL,100/300,2000,1133.85,0,439870,MALE,PhD,priv-house-serv,exercise,not-in-family,60200,0,03-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Springfield,2272 Embaracadero Drive,0,3,YES,2,2,YES,60480,5040,15120,40320,Volkswagen,Jetta,2003,N
446,61,338070,25-01-2006,IN,500/1000,1000,1037.32,0,438837,FEMALE,High School,tech-support,skydiving,wife,0,-15700,30-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NC,Hillsdale,5341 5th Ave,1,3,?,2,1,NO,80880,6740,13480,60660,Nissan,Ultima,2005,N
476,61,865607,18-04-1993,IN,250/500,1000,1562.8,0,458997,FEMALE,Masters,handlers-cleaners,dancing,not-in-family,42800,-68200,18-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,WV,Hillsdale,7745 Washington Ridge,10,1,?,1,2,YES,49390,8980,4490,35920,Suburu,Legacy,2009,N
70,29,963285,09-12-2006,IN,100/300,1000,1425.79,0,604147,FEMALE,MD,armed-forces,video-games,other-relative,62400,-52300,04-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,NC,Riverwood,1275 4th Ridge,12,1,NO,0,3,?,69360,11560,11560,46240,Dodge,RAM,2009,N
233,41,728491,30-08-1997,OH,500/1000,2000,1615.14,0,606638,FEMALE,Associate,tech-support,board-games,other-relative,67100,0,20-01-2015,Vehicle Theft,?,Minor Damage,None,NY,Springfield,4857 Weaver St,6,1,NO,0,1,?,3740,680,680,2380,Chevrolet,Malibu,2011,N
122,33,553436,03-06-1991,IL,250/500,500,1236.5,0,619620,MALE,PhD,other-service,bungie-jumping,husband,0,-48700,12-02-2015,Parked Car,?,Trivial Damage,None,NY,Hillsdale,8211 Sky Hwy,1,1,NO,0,1,NO,5060,460,920,3680,Nissan,Ultima,2003,N
335,48,440616,06-09-1995,IL,500/1000,2000,1017.97,0,441671,FEMALE,MD,machine-op-inspct,chess,wife,59900,-34800,19-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Columbus,8617 Best Ave,21,3,NO,0,0,YES,35860,3260,6520,26080,BMW,X5,2005,Y
257,40,463237,09-02-2000,IN,100/300,2000,1306,0,610381,MALE,Associate,machine-op-inspct,cross-fit,husband,46100,-46900,21-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Ambulance,NY,Columbus,9856 Apache St,3,3,YES,2,1,?,50050,7700,3850,38500,Ford,Fusion,2008,Y
85,27,753452,23-07-1996,IL,500/1000,2000,1174.14,0,602416,MALE,College,priv-house-serv,dancing,unmarried,50400,-61500,02-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Northbend,1951 Best Ave,14,4,YES,0,0,NO,59070,10740,5370,42960,Toyota,Camry,2012,N
133,30,920554,21-09-2005,IN,500/1000,1000,1231.01,0,459562,MALE,College,adm-clerical,board-games,husband,0,-31700,01-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,SC,Riverwood,1824 5th Lane,19,3,NO,1,2,?,28440,3160,3160,22120,Dodge,Neon,2007,N
119,34,594783,30-12-2011,IL,250/500,500,1299.18,0,463271,FEMALE,College,tech-support,hiking,wife,57100,0,08-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,OH,Springfield,7393 Washington St,7,1,YES,2,1,YES,45540,8280,8280,28980,Honda,Civic,1998,Y
169,34,725330,21-07-1996,IN,100/300,500,1469.75,0,458132,FEMALE,JD,sales,reading,not-in-family,0,-57600,16-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,VA,Arlington,1386 Britain St,0,1,?,0,0,YES,38700,7740,3870,27090,Volkswagen,Passat,2012,N
225,39,607259,08-04-1996,OH,250/500,500,1390.72,0,448949,MALE,Masters,tech-support,paintball,other-relative,83900,-52100,20-02-2015,Parked Car,?,Trivial Damage,None,SC,Northbrook,7928 Maple Ridge,6,1,YES,2,1,YES,5830,1060,530,4240,Nissan,Pathfinder,2011,N
84,32,979336,04-03-2001,IL,500/1000,500,1694.09,7000000,603732,FEMALE,Associate,prof-specialty,cross-fit,husband,0,0,30-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,NY,Northbend,1546 Cherokee Ave,0,1,YES,0,2,?,57240,4770,9540,42930,BMW,X6,1995,Y
169,39,865201,19-10-2001,OH,100/300,2000,1140.15,0,608929,MALE,High School,armed-forces,exercise,husband,0,-36800,19-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,VA,Riverwood,2003 2nd Hwy,9,3,YES,0,3,NO,46200,4200,8400,33600,Suburu,Legacy,2015,N
124,32,140977,18-08-2006,IN,100/300,1000,1310.71,0,469875,FEMALE,Masters,farming-fishing,kayaking,wife,29300,0,25-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,NY,Columbus,9418 5th Hwy,23,1,YES,0,1,NO,57700,5770,5770,46160,Toyota,Camry,2003,N
320,48,787351,28-04-2013,IL,250/500,2000,1730.49,7000000,443342,MALE,College,transport-moving,hiking,not-in-family,46300,-41700,24-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbrook,8770 1st Lane,13,3,NO,2,0,NO,56160,4680,9360,42120,Saab,95,1995,N
297,47,272330,29-11-2009,IN,250/500,500,1616.65,7000000,456363,MALE,MD,adm-clerical,movies,unmarried,0,-59500,16-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,WV,Columbus,1087 Flute Drive,0,3,?,0,3,YES,44400,5550,5550,33300,Jeep,Grand Cherokee,1999,N
421,56,728025,15-02-1990,IN,100/300,500,1935.85,4000000,470826,MALE,Masters,machine-op-inspct,reading,own-child,49500,-81100,12-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,NY,Hillsdale,2217 Tree Lane,7,1,?,2,3,?,92730,16860,8430,67440,Mercedes,E400,2004,Y
136,33,804608,12-04-2002,OH,250/500,1000,855.14,0,458582,FEMALE,PhD,craft-repair,paintball,not-in-family,37900,0,04-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,NC,Northbrook,6741 Oak Ridge,23,1,YES,0,1,YES,30700,3070,6140,21490,Toyota,Corolla,2015,N
46,24,718829,21-02-1999,OH,250/500,2000,1568.47,4000000,454480,FEMALE,High School,armed-forces,yachting,unmarried,46800,0,02-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,NY,Northbrook,2123 MLK Ridge,7,1,NO,2,0,?,56600,11320,5660,39620,Toyota,Camry,1999,N
34,24,482404,18-06-2011,IN,500/1000,2000,1550.53,0,435632,FEMALE,MD,armed-forces,dancing,own-child,0,-27700,01-02-2015,Parked Car,?,Trivial Damage,None,VA,Hillsdale,4390 4th Drive,20,1,YES,0,1,?,3960,660,660,2640,Audi,A3,1998,N
95,30,331170,26-03-1995,IL,250/500,2000,1370.92,0,442206,MALE,College,transport-moving,video-games,unmarried,48900,0,14-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,SC,Arlington,1437 3rd Lane,22,1,YES,0,3,?,34800,3480,6960,24360,Accura,MDX,1999,N
140,36,753056,03-05-1991,IN,250/500,500,1363.59,0,468303,FEMALE,JD,armed-forces,kayaking,not-in-family,43200,0,08-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Riverwood,1186 Rock St,10,3,?,2,0,YES,79500,7950,7950,63600,Chevrolet,Tahoe,2000,N
200,34,910365,19-12-2001,IN,250/500,1000,828.42,3000000,467762,FEMALE,College,prof-specialty,basketball,other-relative,0,0,22-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,NY,Arlington,4394 Oak St,10,1,?,2,2,NO,56000,5600,5600,44800,Chevrolet,Malibu,2009,N
123,29,379268,05-08-2012,IN,250/500,500,1209.63,0,447188,FEMALE,Masters,machine-op-inspct,chess,not-in-family,64800,-44200,14-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,NC,Arlington,8368 Cherokee Ave,17,1,YES,1,0,YES,73260,16280,0,56980,Volkswagen,Jetta,2014,Y
267,46,362843,09-08-2004,OH,250/500,2000,1111.17,0,469438,MALE,MD,craft-repair,base-jumping,unmarried,35000,0,03-02-2015,Parked Car,?,Trivial Damage,None,WV,Arlington,4905 Best Lane,3,1,YES,2,3,YES,4950,900,450,3600,Toyota,Camry,1995,N
290,42,135400,20-01-2014,IN,500/1000,500,989.97,0,462519,MALE,Masters,machine-op-inspct,kayaking,own-child,32500,0,20-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Hillsdale,3618 Sky Ave,10,1,NO,0,1,NO,48000,4800,9600,33600,Saab,95,2013,N
45,37,798579,19-12-2011,IN,250/500,1000,1114.23,0,432534,MALE,College,prof-specialty,dancing,wife,0,0,01-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,SC,Arlington,5459 MLK Ave,1,1,YES,0,1,YES,52200,10440,5220,36540,Nissan,Pathfinder,2005,N
186,38,250833,28-07-2008,IN,250/500,2000,1347.31,0,436467,FEMALE,JD,protective-serv,dancing,unmarried,80900,-111100,02-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Springfield,1371 Texas Lane,1,3,NO,2,3,?,73800,12300,12300,49200,Audi,A3,1995,N
135,34,824116,05-05-1998,IL,250/500,2000,1687.53,0,465674,FEMALE,JD,protective-serv,base-jumping,other-relative,0,-69600,23-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,NC,Northbend,2654 Embaracadero St,7,1,?,1,2,NO,78200,15640,7820,54740,Audi,A3,2009,N
110,33,322613,16-04-1995,IN,250/500,1000,1183.48,0,442389,MALE,Associate,other-service,bungie-jumping,husband,0,0,26-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,VA,Northbend,2123 Texas Ave,19,1,?,2,3,NO,55200,13800,9200,32200,Saab,93,2015,Y
259,43,871305,14-02-1992,IL,500/1000,2000,1537.13,0,471614,FEMALE,PhD,handlers-cleaners,kayaking,own-child,0,-58300,02-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,NY,Northbend,4538 Flute Hwy,3,3,NO,0,2,YES,57060,6340,6340,44380,Ford,Fusion,2012,N
114,30,488037,11-07-2007,OH,250/500,1000,1173.25,0,442936,FEMALE,Masters,protective-serv,dancing,husband,0,-34700,25-02-2015,Vehicle Theft,?,Minor Damage,None,WV,Arlington,4434 Weaver St,3,1,NO,0,3,YES,4680,520,520,3640,Chevrolet,Malibu,2013,N
404,56,485813,07-04-2010,IN,250/500,1000,1361.16,4000000,437944,FEMALE,Masters,transport-moving,cross-fit,not-in-family,0,-63700,15-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Ambulance,VA,Hillsdale,2798 1st Ave,23,1,NO,2,0,YES,53100,5310,5310,42480,Accura,MDX,2005,Y
282,48,886473,10-03-1991,OH,500/1000,2000,1422.56,7000000,473705,FEMALE,MD,prof-specialty,video-games,husband,26900,-55300,09-02-2015,Vehicle Theft,?,Minor Damage,None,WV,Springfield,2809 Francis Lane,7,1,?,1,2,NO,3520,640,320,2560,Accura,MDX,2013,N
57,25,907113,20-01-1996,IL,500/1000,2000,1143.06,0,469363,FEMALE,Masters,tech-support,dancing,own-child,63100,-54100,16-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Ambulance,VA,Riverwood,7281 Oak St,0,3,NO,0,1,YES,72900,14580,14580,43740,Nissan,Maxima,2010,N
215,38,833321,01-03-2010,IN,250/500,500,1405.71,0,465376,FEMALE,PhD,craft-repair,camping,unmarried,0,0,01-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,SC,Arlington,9878 Washington Ave,10,1,?,0,1,NO,70700,7070,14140,49490,Volkswagen,Passat,2008,N
140,30,521592,15-06-2014,IL,100/300,500,1354.2,0,438775,FEMALE,College,adm-clerical,bungie-jumping,wife,100500,0,10-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,SC,Columbus,2537 5th Ave,4,4,?,0,0,?,60170,5470,10940,43760,Nissan,Pathfinder,2006,N
250,42,254837,25-11-2004,IN,100/300,500,1055.6,0,457962,MALE,High School,exec-managerial,paintball,husband,69500,-40700,03-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,SC,Columbus,8493 Apache Drive,16,1,?,1,1,?,74800,13600,6800,54400,Ford,Fusion,2009,Y
286,41,634499,26-08-2000,IL,250/500,1000,999.43,0,477947,MALE,College,prof-specialty,paintball,wife,25800,0,01-01-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Northbend,2878 Britain Hwy,3,1,YES,2,0,?,4100,820,410,2870,Chevrolet,Malibu,2009,N
356,47,574707,23-08-2005,IN,250/500,2000,1155.97,0,431104,MALE,High School,prof-specialty,camping,husband,0,0,23-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,SC,Columbus,2862 Tree Ridge,5,1,YES,0,3,?,61490,5590,11180,44720,Dodge,RAM,2009,N
65,29,476839,09-08-1990,IL,250/500,1000,1726.91,0,456570,MALE,High School,other-service,basketball,own-child,0,0,28-01-2015,Vehicle Theft,?,Trivial Damage,None,VA,Hillsdale,4453 Best Ave,14,1,?,0,0,?,7200,720,1440,5040,Audi,A5,1999,N
187,34,149601,28-03-2003,IN,500/1000,500,1232.57,0,612986,FEMALE,PhD,machine-op-inspct,polo,not-in-family,59500,0,22-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,NY,Arlington,5191 4th St,7,3,?,0,0,?,45100,8200,4100,32800,Nissan,Pathfinder,2011,N
386,53,630683,23-10-2007,OH,250/500,500,1078.65,0,615730,MALE,JD,craft-repair,camping,not-in-family,36800,0,03-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,WV,Northbend,1364 Best St,16,1,?,2,3,YES,66660,12120,6060,48480,Honda,Civic,2006,N
197,41,500639,27-06-1996,OH,500/1000,1000,1324.78,0,478640,FEMALE,PhD,prof-specialty,basketball,not-in-family,0,-64500,10-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Police,NC,Northbrook,8946 2nd Drive,6,1,NO,2,2,YES,76400,15280,7640,53480,Volkswagen,Jetta,1997,Y
166,37,352120,11-12-1994,IN,250/500,500,1518.54,0,470510,FEMALE,MD,craft-repair,kayaking,not-in-family,0,0,25-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Ambulance,PA,Riverwood,3726 MLK Hwy,10,1,YES,1,1,NO,58300,10600,10600,37100,Ford,F150,2001,N
293,49,569245,05-12-1995,IL,100/300,2000,1239.06,0,439360,FEMALE,JD,transport-moving,skydiving,husband,34900,0,10-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,SC,Riverwood,2820 Britain St,19,1,?,1,1,?,57310,5210,10420,41680,Volkswagen,Passat,2002,N
179,32,907012,15-12-1996,OH,500/1000,2000,1246.68,0,440251,FEMALE,PhD,priv-house-serv,movies,own-child,0,0,28-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NY,Arlington,2646 MLK Drive,10,1,?,0,1,?,53100,5900,5900,41300,Suburu,Impreza,2006,N
76,24,700074,06-06-2011,OH,250/500,1000,1622.67,0,600313,FEMALE,MD,priv-house-serv,paintball,husband,0,0,18-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Ambulance,WV,Riverwood,6256 Elm St,6,3,NO,1,1,?,74700,14940,7470,52290,Suburu,Forrestor,1997,N
105,28,866805,13-12-1995,OH,250/500,500,1082.36,0,452216,FEMALE,Associate,prof-specialty,golf,own-child,0,0,24-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Riverwood,9724 Maple St,12,3,NO,2,2,NO,60500,12100,6050,42350,Audi,A5,1995,N
97,26,951863,28-10-1997,OH,250/500,1000,1270.55,0,478532,MALE,Masters,protective-serv,chess,unmarried,0,-72100,20-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,NC,Riverwood,7397 4th Drive,10,3,YES,2,3,NO,84920,7720,15440,61760,Jeep,Wrangler,2006,Y
148,36,211578,04-01-1996,IL,500/1000,1000,1236.32,5000000,616929,FEMALE,MD,adm-clerical,hiking,own-child,55100,0,10-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,PA,Northbrook,3488 Flute Lane,0,1,NO,2,0,NO,61050,5550,11100,44400,Dodge,Neon,2009,N
77,26,357394,09-05-2008,IL,250/500,2000,785.82,0,620207,MALE,JD,exec-managerial,movies,other-relative,49700,0,07-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NY,Hillsdale,9082 3rd Lane,12,3,YES,2,0,?,69080,12560,6280,50240,Audi,A5,2009,Y
295,46,863749,05-12-2009,IN,250/500,500,1265.84,0,605743,FEMALE,JD,prof-specialty,paintball,own-child,52200,-44500,16-01-2015,Vehicle Theft,?,Minor Damage,None,VA,Arlington,1941 5th Ridge,10,1,YES,1,3,YES,4560,760,380,3420,Nissan,Pathfinder,2007,N
126,28,596914,05-01-1992,IN,250/500,500,1508.9,0,472814,FEMALE,JD,machine-op-inspct,skydiving,other-relative,0,0,01-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,WV,Northbend,5333 MLK Lane,3,4,NO,0,1,?,67800,11300,11300,45200,Ford,F150,2011,N
132,32,684653,15-11-1997,OH,250/500,2000,1106.84,0,464362,MALE,PhD,other-service,reading,wife,43100,-31900,07-01-2015,Vehicle Theft,?,Minor Damage,Police,SC,Riverwood,4577 Sky Hwy,21,1,YES,1,1,YES,5600,1120,560,3920,Suburu,Legacy,2005,N
370,55,528259,22-12-2012,IN,500/1000,2000,1389.13,7000000,456203,MALE,JD,other-service,basketball,wife,0,-53200,17-02-2015,Vehicle Theft,?,Minor Damage,None,NC,Northbrook,4814 Lincoln Lane,6,1,?,0,2,?,9000,900,1800,6300,Mercedes,ML350,2015,N
257,43,797636,19-05-1992,IN,100/300,1000,974.84,0,468984,FEMALE,JD,transport-moving,kayaking,other-relative,52100,0,26-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,VA,Northbend,2381 1st Hwy,0,1,NO,0,1,YES,85320,21330,7110,56880,Nissan,Pathfinder,2006,N
9,24,326180,25-05-2002,IL,100/300,2000,1304.46,0,473349,FEMALE,PhD,machine-op-inspct,golf,other-relative,51700,-33300,31-01-2015,Vehicle Theft,?,Trivial Damage,None,NC,Arlington,6939 3rd Hwy,6,1,NO,0,3,YES,5940,540,1080,4320,Audi,A5,2001,Y
185,34,620075,21-04-2010,OH,250/500,500,1257.36,0,474771,FEMALE,PhD,armed-forces,movies,husband,0,0,27-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,SC,Arlington,5269 Flute Hwy,20,3,YES,1,1,?,51370,9340,4670,37360,BMW,M5,2000,Y
234,43,965187,26-03-1990,OH,250/500,500,1257.04,0,448294,MALE,Associate,protective-serv,reading,own-child,0,-48800,01-03-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,SC,Northbrook,7197 2nd Drive,4,1,NO,2,2,YES,51600,10320,5160,36120,Dodge,Neon,2011,N
253,44,516182,12-05-2007,OH,100/300,2000,719.52,0,606606,FEMALE,High School,farming-fishing,golf,own-child,45800,0,13-02-2015,Parked Car,?,Trivial Damage,None,SC,Arlington,1741 Best Ridge,9,1,NO,0,3,?,5400,600,600,4200,Chevrolet,Malibu,1998,N
233,39,728839,02-01-2001,OH,500/1000,2000,1524.18,0,605220,MALE,JD,craft-repair,reading,unmarried,0,0,08-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Fire,SC,Northbrook,9148 4th Hwy,20,3,?,1,0,YES,48870,5430,5430,38010,Saab,95,1999,N
274,44,771509,10-08-2006,IN,500/1000,500,1395.58,0,466612,FEMALE,JD,tech-support,reading,husband,0,0,05-02-2015,Vehicle Theft,?,Minor Damage,Police,WV,Springfield,4279 Solo Drive,7,1,NO,2,1,?,5590,860,860,3870,BMW,X5,2000,N
297,48,264221,28-07-2014,IL,500/1000,1000,1243.68,0,463331,MALE,Masters,protective-serv,camping,wife,0,-71400,20-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Other,NY,Springfield,9177 Texas Ave,18,3,?,0,2,?,54960,6870,0,48090,Toyota,Corolla,2002,Y
273,47,602704,27-09-2011,OH,500/1000,1000,1189.04,0,457843,FEMALE,Associate,prof-specialty,video-games,own-child,59600,0,24-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,VA,Northbrook,5969 Francis St,0,3,NO,1,3,?,39800,7960,3980,27860,Jeep,Wrangler,2014,N
147,37,672416,20-04-2013,IN,500/1000,2000,1375.29,0,609226,FEMALE,Masters,armed-forces,chess,own-child,0,0,31-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,SC,Columbus,9942 Tree Ave,11,3,?,0,1,?,56160,6240,6240,43680,Ford,Fusion,2015,Y
285,42,545506,20-03-1991,IN,100/300,500,1389.13,0,452942,MALE,Associate,priv-house-serv,golf,not-in-family,63100,-79400,23-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,VA,Hillsdale,5474 Weaver Hwy,13,1,NO,0,3,?,52700,5270,10540,36890,Toyota,Corolla,2005,N
289,43,777533,21-12-2002,OH,500/1000,1000,1387.51,0,609390,FEMALE,Associate,sales,base-jumping,not-in-family,0,0,11-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,NY,Riverwood,1102 Apache Hwy,19,3,YES,1,3,?,68580,7620,7620,53340,Jeep,Wrangler,2010,N
427,60,953334,03-12-2005,IN,100/300,1000,1178.61,7000000,446608,MALE,MD,craft-repair,board-games,own-child,0,-54400,20-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Springfield,9214 Texas Drive,23,1,YES,1,2,YES,90860,12980,19470,58410,Volkswagen,Jetta,2004,Y
380,53,369781,25-05-2011,IL,250/500,2000,1166.62,6000000,602500,MALE,Associate,priv-house-serv,bungie-jumping,wife,0,0,24-02-2015,Parked Car,?,Trivial Damage,Police,NC,Northbend,8991 Texas Hwy,23,1,NO,0,3,NO,5700,570,570,4560,Saab,93,2001,N
13,21,990998,18-10-2006,IN,100/300,1000,1556.31,0,463809,MALE,Associate,prof-specialty,golf,not-in-family,0,-75000,19-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,NY,Hillsdale,9580 MLK Ave,19,3,YES,2,0,YES,94930,8630,8630,77670,Accura,RSX,2014,N
282,43,982678,19-07-2006,OH,250/500,500,1452.27,0,611996,MALE,MD,farming-fishing,video-games,not-in-family,75800,0,08-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,SC,Northbend,5868 Best Drive,19,1,NO,1,2,NO,46800,4680,9360,32760,Audi,A5,2007,Y
312,47,646069,08-06-2002,OH,500/1000,1000,1212.07,0,459298,FEMALE,MD,exec-managerial,polo,wife,66900,-51800,01-03-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,NY,Northbend,5318 5th Ave,17,3,NO,2,3,NO,56320,7040,7040,42240,Mercedes,ML350,2000,N
266,46,331683,12-02-2009,OH,100/300,2000,1578.54,0,468158,MALE,Associate,handlers-cleaners,chess,husband,0,-41400,21-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Hillsdale,7502 Rock Lane,18,1,NO,1,3,YES,83490,7590,15180,60720,Nissan,Pathfinder,1996,N
30,36,364055,14-05-2001,IN,500/1000,500,1488.26,0,440831,FEMALE,College,machine-op-inspct,golf,wife,0,-63500,28-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,WV,Northbrook,4627 Elm Ridge,17,3,NO,2,2,?,57900,5790,5790,46320,Saab,95,2008,N
198,36,521854,16-02-2001,IN,250/500,1000,1096.39,0,603848,MALE,High School,armed-forces,kayaking,own-child,0,0,26-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,SC,Springfield,5584 Britain Lane,11,3,?,1,3,YES,49410,5490,5490,38430,Audi,A3,2015,N
290,45,737252,18-11-1993,OH,500/1000,2000,1215.36,0,617739,MALE,Associate,tech-support,reading,own-child,54400,0,31-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,WV,Northbend,7002 Oak Hwy,22,3,?,0,1,NO,66200,6620,6620,52960,Suburu,Impreza,2012,N
260,46,344480,18-02-1990,OH,100/300,2000,1482.57,0,607133,MALE,MD,priv-house-serv,reading,husband,35000,0,20-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NY,Columbus,4780 Best Drive,7,1,NO,0,1,NO,64080,10680,10680,42720,Toyota,Camry,2005,N
233,43,898519,21-05-2000,OH,250/500,1000,954.18,0,437470,FEMALE,College,tech-support,dancing,other-relative,0,0,17-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,SC,Northbrook,8995 1st Ave,17,3,?,2,3,YES,42500,8500,4250,29750,Nissan,Pathfinder,2000,N
130,30,957816,26-08-2012,IL,500/1000,2000,1193.4,0,461372,MALE,PhD,exec-managerial,bungie-jumping,own-child,0,-40800,02-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,SC,Columbus,5586 2nd St,16,3,NO,2,3,?,48950,8900,4450,35600,Suburu,Legacy,2005,N
230,42,175960,16-11-2004,IN,100/300,1000,1023.11,0,476130,FEMALE,MD,adm-clerical,golf,own-child,0,-45300,06-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Northbend,1589 Best Ave,13,3,NO,1,2,YES,58850,10700,10700,37450,Accura,MDX,1999,N
212,40,489618,23-01-2003,IL,500/1000,1000,1524.45,0,452438,FEMALE,Masters,other-service,golf,husband,73200,0,11-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,NY,Northbrook,1880 Weaver Drive,17,3,YES,0,2,YES,82400,8240,8240,65920,Nissan,Pathfinder,2006,N
299,44,717044,07-11-2008,OH,500/1000,1000,1653.32,0,460517,FEMALE,College,other-service,bungie-jumping,other-relative,0,-48800,25-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Springfield,7295 Tree Hwy,3,1,YES,2,0,?,54240,6780,6780,40680,Suburu,Impreza,2009,N
91,26,101421,19-10-1999,IL,250/500,1000,1022.46,0,444896,FEMALE,Associate,armed-forces,video-games,other-relative,52700,0,23-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,SC,Arlington,8832 Pine Drive,15,3,?,0,2,?,74200,7420,7420,59360,Jeep,Wrangler,1996,N
398,53,793948,20-12-1990,IL,100/300,2000,1396.43,0,448722,FEMALE,Associate,priv-house-serv,base-jumping,unmarried,21500,0,29-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,WV,Hillsdale,1620 Oak Ave,16,1,YES,2,1,?,47430,5270,5270,36890,Toyota,Camry,2000,N
218,43,737483,14-02-1996,IL,250/500,500,1521.55,0,477856,FEMALE,Associate,priv-house-serv,polo,other-relative,61100,-64500,02-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,SC,Hillsdale,3847 Elm Hwy,18,1,?,1,3,YES,68200,13640,6820,47740,Dodge,RAM,2003,Y
152,33,695117,10-06-2001,IN,100/300,1000,1034.27,0,617721,FEMALE,JD,armed-forces,exercise,husband,0,0,06-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NY,Hillsdale,3177 MLK Ridge,18,1,NO,1,0,NO,63900,7100,7100,49700,Accura,TL,2014,N
212,39,167466,17-03-2010,OH,100/300,1000,1255.35,0,454176,FEMALE,JD,protective-serv,skydiving,not-in-family,60300,-58900,14-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Ambulance,NY,Riverwood,3929 Oak Drive,22,2,NO,0,3,YES,59300,11860,5930,41510,Dodge,RAM,2008,N
242,44,664732,30-07-2003,IL,500/1000,2000,1396.89,6000000,618127,FEMALE,College,priv-house-serv,chess,other-relative,0,-61600,04-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Other,SC,Springfield,1469 Lincoln Drive,15,3,NO,1,2,YES,66900,6690,13380,46830,Suburu,Forrestor,1999,Y
80,27,143038,17-09-2014,OH,500/1000,500,795.31,0,441923,MALE,JD,farming-fishing,skydiving,husband,0,-51000,12-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,OH,Arlington,9719 4th Lane,16,3,YES,1,3,?,40810,3710,7420,29680,Ford,F150,2000,Y
260,43,979963,03-06-2009,IN,100/300,500,982.22,0,604279,FEMALE,JD,exec-managerial,skydiving,not-in-family,54500,-72100,12-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Riverwood,3196 Cherokee St,18,1,?,1,0,?,75400,15080,7540,52780,Honda,CRV,2011,N
133,34,467841,11-10-1994,IN,500/1000,500,1074.07,0,440833,FEMALE,JD,prof-specialty,bungie-jumping,husband,70900,-61100,28-01-2015,Parked Car,?,Minor Damage,None,WV,Northbend,8492 Andromedia Ridge,8,1,NO,2,0,YES,4200,420,840,2940,Jeep,Wrangler,2013,N
290,45,219028,18-07-1991,OH,100/300,1000,1311.3,0,451550,FEMALE,Associate,machine-op-inspct,chess,wife,38500,0,10-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,NY,Hillsdale,1353 Washington St,23,3,YES,0,0,YES,52650,5850,5850,40950,Ford,F150,2001,Y
322,49,130156,24-09-2001,IL,250/500,2000,1277.12,0,431853,FEMALE,PhD,armed-forces,kayaking,own-child,0,-46000,19-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,WV,Hillsdale,6731 Andromedia Hwy,18,1,?,0,2,YES,42240,7680,7680,26880,Chevrolet,Malibu,2007,N
228,39,762951,19-09-2012,IN,500/1000,500,1388.62,0,614274,FEMALE,JD,sales,reading,husband,35200,0,19-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,NC,Riverwood,5769 Texas Lane,10,1,YES,1,0,YES,59490,6610,6610,46270,Mercedes,ML350,1995,N
195,37,376879,11-07-1991,IL,100/300,1000,1406.52,8000000,619148,MALE,MD,tech-support,base-jumping,unmarried,0,0,28-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Ambulance,SC,Arlington,2849 Pine Drive,12,1,NO,0,2,NO,44200,4420,4420,35360,Jeep,Wrangler,2002,Y
247,39,599031,29-10-1991,IN,100/300,500,1558.29,0,456781,FEMALE,Masters,protective-serv,reading,unmarried,0,-49300,16-02-2015,Vehicle Theft,?,Trivial Damage,Police,WV,Springfield,2577 Texas Ridge,5,1,YES,1,2,?,7700,770,1540,5390,Saab,93,2000,N
405,57,676255,28-12-1999,IN,500/1000,1000,1132.47,4000000,434293,MALE,MD,priv-house-serv,exercise,other-relative,46300,0,08-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,NY,Northbend,3841 Washington Lane,21,3,?,2,3,?,61440,10240,10240,40960,Saab,93,2008,N
144,37,985446,11-10-2012,OH,250/500,2000,1896.91,0,460895,FEMALE,PhD,handlers-cleaners,sleeping,not-in-family,73700,0,24-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Fire,SC,Columbus,8125 Texas Ridge,17,3,YES,1,3,NO,54400,5440,10880,38080,Chevrolet,Silverado,2015,Y
338,47,884180,19-08-1995,IL,500/1000,500,1143.46,4000000,601600,MALE,MD,priv-house-serv,polo,other-relative,0,0,18-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NC,Arlington,4826 5th St,4,1,YES,2,1,?,58560,9760,9760,39040,Mercedes,E400,2002,N
121,34,571462,11-02-1991,IN,500/1000,500,1285.42,0,465440,MALE,MD,priv-house-serv,video-games,wife,0,0,21-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Fire,VA,Northbrook,1578 5th Lane,11,1,NO,1,1,NO,67300,6730,6730,53840,Dodge,RAM,2000,Y
398,55,815883,02-07-1991,OH,250/500,2000,1305.26,0,455482,MALE,MD,farming-fishing,skydiving,wife,66200,-49700,08-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,VA,Hillsdale,6440 Rock Lane,18,1,?,1,3,YES,36740,3340,6680,26720,BMW,X5,1998,Y
9,30,258265,10-04-1994,IL,100/300,1000,1073.83,0,438877,FEMALE,High School,machine-op-inspct,dancing,not-in-family,0,0,02-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,NY,Northbrook,5806 Embaracadero St,12,1,?,0,0,NO,85690,15580,15580,54530,BMW,3 Series,2011,N
115,31,569714,04-12-2005,OH,500/1000,1000,1051.67,0,479824,FEMALE,Associate,exec-managerial,bungie-jumping,not-in-family,0,0,01-03-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,WV,Riverwood,1472 4th Drive,18,3,YES,0,3,NO,34160,0,4270,29890,Audi,A5,2005,Y
280,48,180008,16-07-2014,IL,250/500,1000,1387.35,0,477415,MALE,JD,transport-moving,paintball,not-in-family,0,-72000,04-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,WV,Hillsdale,5839 Weaver Lane,16,1,NO,2,2,?,61320,10220,10220,40880,BMW,M5,1998,N
254,45,633375,17-09-2003,IL,250/500,500,1083.64,0,614372,MALE,JD,other-service,paintball,husband,59800,0,27-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,NC,Columbus,7630 Rock Drive,19,3,YES,0,0,?,79680,13280,13280,53120,BMW,3 Series,2004,N
141,30,556538,15-07-2000,IL,250/500,1000,1851.78,0,465248,FEMALE,High School,craft-repair,exercise,other-relative,78800,0,09-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,SC,Riverwood,7144 Andromedia St,13,1,NO,1,0,YES,61740,6860,6860,48020,Audi,A3,2002,N
441,55,669501,29-07-2009,IN,250/500,500,1270.29,4000000,449421,MALE,College,armed-forces,exercise,husband,24000,-50500,19-02-2015,Parked Car,?,Minor Damage,None,VA,Arlington,9988 Rock Ridge,4,1,NO,0,0,NO,6400,640,640,5120,Honda,Civic,2002,N
381,55,963761,13-04-1991,OH,500/1000,500,1459.99,0,445856,FEMALE,MD,other-service,chess,wife,35900,0,12-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,SC,Northbrook,7544 Washington Ave,8,1,YES,1,2,YES,60600,12120,6060,42420,Accura,TL,2011,N
191,38,753005,20-11-2005,IL,100/300,2000,1253.44,0,608525,FEMALE,Masters,craft-repair,sleeping,not-in-family,0,0,07-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,VA,Hillsdale,7201 Washington Ave,19,3,NO,2,0,NO,56320,10240,5120,40960,Volkswagen,Jetta,2007,N
145,34,454758,20-05-1990,IN,100/300,1000,1142.48,0,608813,FEMALE,JD,priv-house-serv,sleeping,other-relative,0,0,13-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,NC,Northbend,8805 Cherokee Drive,18,1,YES,2,0,NO,52250,9500,4750,38000,Suburu,Legacy,2012,N
479,60,698589,28-11-2002,IL,500/1000,1000,1188.45,0,459295,FEMALE,MD,exec-managerial,camping,other-relative,0,-44800,18-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,SC,Arlington,3275 Pine St,9,2,?,0,3,?,53900,5390,10780,37730,Saab,95,2006,N
215,35,330119,15-06-2004,IL,500/1000,1000,1125.4,0,606144,MALE,Masters,adm-clerical,yachting,husband,40000,-43400,15-01-2015,Vehicle Theft,?,Trivial Damage,None,WV,Columbus,7785 Lincoln Lane,6,1,?,2,1,NO,2640,220,440,1980,Jeep,Wrangler,2001,N
41,33,164464,26-09-2010,OH,250/500,500,1294.41,0,476315,MALE,High School,transport-moving,sleeping,husband,0,0,23-02-2015,Vehicle Theft,?,Minor Damage,None,NC,Arlington,4994 Lincoln Drive,8,1,YES,0,0,?,8970,1380,1380,6210,Dodge,Neon,2011,N
45,31,927354,15-09-1990,IN,100/300,500,1459.5,0,475891,MALE,MD,priv-house-serv,movies,not-in-family,0,0,17-02-2015,Parked Car,?,Minor Damage,None,NY,Springfield,1298 Maple Hwy,6,1,?,1,3,?,6000,1000,1000,4000,Suburu,Impreza,2000,N
156,38,231508,16-09-2009,IL,100/300,500,1367.99,0,462525,MALE,High School,armed-forces,board-games,own-child,26500,0,17-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,WV,Northbend,2644 MLK Drive,23,3,?,0,3,?,55200,11040,5520,38640,Saab,92x,1998,Y
246,45,272910,12-08-1999,IN,250/500,500,1594.37,0,606283,MALE,Associate,exec-managerial,board-games,own-child,0,0,18-01-2015,Parked Car,?,Trivial Damage,None,SC,Riverwood,5630 1st Drive,13,1,NO,0,3,YES,7260,660,1320,5280,Saab,92x,2008,N
178,39,305758,08-03-2009,IL,100/300,500,1035.99,0,465252,FEMALE,JD,exec-managerial,sleeping,own-child,0,0,17-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NY,Springfield,6137 MLK St,3,3,NO,2,0,?,64680,11760,11760,41160,Jeep,Wrangler,2010,N
237,43,950542,27-04-2009,OH,250/500,500,911.53,0,449979,FEMALE,PhD,sales,basketball,husband,53200,0,25-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,WV,Northbrook,5383 Maple Drive,23,1,NO,0,1,NO,59200,0,11840,47360,Chevrolet,Malibu,1998,N
127,34,291544,02-08-2006,OH,500/1000,500,1319.97,0,604681,FEMALE,Associate,craft-repair,paintball,own-child,73700,0,06-01-2015,Vehicle Theft,?,Minor Damage,None,NC,Arlington,4460 4th Lane,8,1,YES,1,3,?,4700,470,940,3290,Saab,92x,1998,N
1,33,388616,06-12-1995,OH,250/500,2000,1391.63,0,466390,MALE,Associate,sales,video-games,husband,61200,0,26-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Columbus,8524 Pine Lane,23,1,YES,0,3,NO,69400,6940,6940,55520,Mercedes,C300,2000,N
5,21,577992,13-11-2002,IN,250/500,500,915.41,5000000,612316,FEMALE,High School,exec-managerial,sleeping,own-child,0,0,11-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,NY,Northbrook,8456 1st Ave,23,1,YES,0,0,NO,40500,4050,4050,32400,Nissan,Pathfinder,1998,N
64,28,342830,09-11-1991,IL,500/1000,1000,1468.82,0,474731,MALE,JD,handlers-cleaners,skydiving,other-relative,56800,-51800,13-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Fire,SC,Riverwood,3639 Flute Hwy,9,3,NO,2,1,NO,60000,5000,10000,45000,Honda,Accord,1997,N
142,30,491170,14-01-1998,IN,500/1000,500,1412.76,0,603260,MALE,PhD,armed-forces,basketball,wife,66400,-63700,10-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,WV,Riverwood,7900 Sky Hwy,22,1,YES,2,3,NO,67320,11220,11220,44880,Volkswagen,Jetta,1996,N
97,27,175553,25-04-2002,OH,500/1000,500,1588.26,0,434370,FEMALE,High School,tech-support,movies,husband,56700,-49300,24-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,VA,Riverwood,7835 Cherokee Hwy,22,3,YES,2,1,YES,75690,8410,8410,58870,Saab,95,2014,N
121,31,439341,20-07-1991,IN,100/300,1000,1140.91,0,478388,MALE,Associate,adm-clerical,paintball,other-relative,51300,0,15-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,VA,Northbend,1030 Pine Lane,15,3,NO,1,2,?,64300,6430,6430,51440,Chevrolet,Silverado,2002,Y
225,43,221186,13-08-2004,OH,100/300,1000,1517.54,0,617883,MALE,JD,priv-house-serv,camping,own-child,0,-20900,09-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,SC,Columbus,9278 Francis Ridge,16,3,NO,2,0,YES,64400,6440,6440,51520,BMW,X5,2011,N
425,53,868031,24-06-1990,OH,250/500,2000,912.29,0,464808,FEMALE,High School,priv-house-serv,paintball,husband,42900,-39100,31-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Fire,WV,Northbend,6604 Apache Drive,17,3,?,1,2,?,97080,16180,16180,64720,Saab,92x,2005,N
285,44,844117,21-08-1991,OH,250/500,2000,1144.3,0,609458,MALE,MD,priv-house-serv,base-jumping,not-in-family,52600,0,04-02-2015,Vehicle Theft,?,Minor Damage,Police,WV,Northbrook,2311 4th St,3,1,YES,1,0,?,5500,500,500,4500,Honda,Civic,2010,N
192,38,744557,25-02-2011,IN,500/1000,1000,1281.43,0,432405,FEMALE,Masters,transport-moving,skydiving,not-in-family,65100,-50300,30-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Other,SC,Northbend,9523 Solo Hwy,10,3,?,0,2,NO,30700,3070,6140,21490,Jeep,Wrangler,2010,N
285,48,159536,04-02-2013,IL,100/300,2000,1101.85,0,457875,FEMALE,College,sales,dancing,wife,46100,0,27-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Police,PA,Springfield,3171 Andromedia Lane,9,1,NO,1,2,YES,33480,3720,3720,26040,Nissan,Pathfinder,2012,N
98,26,727109,20-02-2001,IN,500/1000,2000,1082.1,0,477268,MALE,JD,exec-managerial,kayaking,other-relative,0,-30900,05-02-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,WV,Northbend,2492 Lincoln Lane,13,2,?,0,1,YES,65430,14540,7270,43620,Jeep,Wrangler,2001,N
175,36,155604,03-03-1992,OH,500/1000,500,1185.44,0,437580,MALE,Masters,exec-managerial,kayaking,not-in-family,44900,-52500,30-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,NC,Northbend,4477 5th Ave,15,3,YES,2,1,YES,42680,3880,7760,31040,Accura,RSX,2006,Y
259,45,608443,21-12-2006,IL,500/1000,2000,1175.07,0,457121,MALE,MD,craft-repair,movies,not-in-family,30100,0,03-01-2015,Single Vehicle Collision,Side Collision,Minor Damage,Fire,WV,Springfield,6724 Andromedia St,23,1,?,1,1,NO,87780,7980,7980,71820,Honda,CRV,2011,N
140,36,117862,14-07-2000,OH,500/1000,2000,979.26,0,436364,FEMALE,JD,transport-moving,cross-fit,own-child,0,-67000,01-03-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,NY,Riverwood,7495 Washington Ave,2,4,YES,0,2,YES,72800,14560,14560,43680,Honda,Accord,1998,N
231,37,991553,12-12-1991,OH,250/500,500,920.81,0,467654,FEMALE,High School,sales,chess,wife,0,0,13-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,SC,Hillsdale,4291 Sky Hwy,14,1,YES,2,0,?,71190,0,7910,63280,Mercedes,C300,1997,Y
186,38,727443,01-07-2013,OH,100/300,500,922.85,0,471148,MALE,High School,adm-clerical,golf,husband,70300,-70900,25-02-2015,Vehicle Theft,?,Trivial Damage,None,NY,Hillsdale,5650 Rock Ave,7,1,?,1,1,YES,3600,400,400,2800,Honda,Civic,1999,N
229,41,378587,16-12-1998,OH,250/500,2000,1107.59,3000000,468202,MALE,PhD,tech-support,chess,not-in-family,65400,0,23-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NY,Columbus,6888 Elm Ridge,23,1,NO,1,3,NO,62640,10440,10440,41760,Mercedes,C300,2009,N
180,36,420948,03-01-2015,IL,100/300,500,1272.46,0,456959,MALE,College,prof-specialty,exercise,wife,0,0,19-02-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Ambulance,NY,Northbrook,2352 Sky Drive,7,3,YES,2,1,?,69630,12660,6330,50640,Toyota,Corolla,1998,N
188,33,457188,01-04-1994,IL,250/500,1000,1340.24,0,447274,MALE,High School,protective-serv,chess,own-child,0,-68800,08-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Police,NY,Hillsdale,5280 Pine Ave,8,3,YES,1,0,?,76010,13820,6910,55280,Dodge,RAM,1995,Y
214,40,118236,15-08-2000,OH,100/300,1000,1648,0,608405,MALE,JD,transport-moving,base-jumping,not-in-family,57700,-43500,04-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,WV,Northbrook,6638 Tree Drive,17,1,NO,1,0,YES,44220,8040,4020,32160,Accura,MDX,2000,N
178,38,987524,13-11-2014,IL,250/500,500,1381.14,0,472253,FEMALE,College,other-service,camping,wife,0,0,22-02-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NY,Northbrook,5678 Lincoln Drive,10,3,NO,0,3,NO,57200,5200,10400,41600,BMW,M5,2011,N
55,35,490596,04-02-2011,IL,500/1000,500,1198.44,8000000,438923,MALE,MD,priv-house-serv,polo,wife,0,0,14-01-2015,Vehicle Theft,?,Minor Damage,Police,WV,Columbus,4496 Pine Lane,9,1,NO,0,3,NO,3080,560,560,1960,Nissan,Ultima,1998,N
90,31,524215,24-06-1990,OH,250/500,2000,951.27,0,607131,FEMALE,PhD,other-service,hiking,not-in-family,42100,0,06-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,SC,Hillsdale,8845 5th Ave,2,1,YES,1,0,YES,75790,13780,6890,55120,Accura,RSX,2007,N
135,30,913464,21-01-2009,IN,500/1000,2000,1341.24,0,601701,FEMALE,MD,farming-fishing,skydiving,wife,37100,-46500,19-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Riverwood,9317 Apache Ave,18,3,NO,0,1,NO,32670,5940,2970,23760,Honda,Accord,2003,N
277,46,398484,07-11-1992,IL,250/500,2000,1177.57,0,469220,FEMALE,Associate,adm-clerical,video-games,husband,0,-65500,24-01-2015,Vehicle Theft,?,Minor Damage,Police,VA,Arlington,8638 3rd Ave,4,1,?,2,3,NO,3870,430,860,2580,Jeep,Wrangler,2010,N
211,38,752504,15-05-1997,IN,250/500,1000,1055.09,0,433250,FEMALE,Masters,transport-moving,video-games,own-child,0,0,27-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,NY,Columbus,3061 Francis Hwy,12,1,?,0,3,YES,91520,8320,16640,66560,BMW,X6,2005,Y
156,32,449263,20-03-1992,IL,250/500,500,1479.48,0,444413,MALE,Masters,prof-specialty,bungie-jumping,unmarried,0,0,13-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Northbrook,1173 Andromedia Ave,15,1,YES,1,3,YES,74690,6790,13580,54320,Dodge,RAM,2008,Y
84,30,844007,17-07-1995,IN,500/1000,2000,1827.38,0,433593,MALE,Associate,priv-house-serv,polo,other-relative,0,-15900,15-01-2015,Vehicle Theft,?,Trivial Damage,Police,VA,Springfield,6068 2nd St,9,1,YES,1,3,?,4620,420,840,3360,Audi,A5,1998,N
136,32,686522,27-12-2000,IN,100/300,500,1169.62,0,458143,FEMALE,JD,sales,yachting,not-in-family,0,0,04-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Police,SC,Arlington,7937 Weaver Ridge,6,1,YES,0,0,NO,55000,10000,10000,35000,Toyota,Camry,2008,Y
310,48,670142,06-08-1999,IN,100/300,500,1516.34,0,474167,FEMALE,JD,adm-clerical,sleeping,unmarried,11000,0,04-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Police,SC,Springfield,2823 Weaver Lane,11,4,YES,0,2,NO,59400,13200,6600,39600,Saab,93,1996,Y
123,34,607687,03-03-2007,OH,500/1000,2000,1270.21,0,476413,FEMALE,College,sales,sleeping,husband,16100,-61200,14-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Fire,PA,Columbus,1809 Sky St,13,3,NO,1,1,?,55260,6140,0,49120,Nissan,Ultima,2000,N
243,44,967713,25-12-1997,IL,250/500,500,809.11,0,600208,MALE,JD,craft-repair,polo,other-relative,33200,0,27-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Other,NC,Springfield,9352 Washington Ave,4,3,?,2,1,YES,51400,5140,10280,35980,Honda,Civic,1996,N
36,37,291902,06-11-2013,IL,500/1000,1000,1115.81,0,618926,FEMALE,Masters,machine-op-inspct,reading,husband,0,-59800,12-02-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Ambulance,NY,Columbus,2697 Oak Drive,20,3,YES,1,3,?,48780,5420,10840,32520,Dodge,Neon,2008,N
146,31,149839,21-09-1990,OH,100/300,1000,1457.65,5000000,606219,FEMALE,College,armed-forces,camping,own-child,0,0,03-02-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Ambulance,VA,Riverwood,1110 4th Drive,0,3,NO,1,3,?,52380,5820,5820,40740,Toyota,Highlander,2010,N
154,34,840225,05-10-1999,OH,100/300,1000,1041.36,0,448436,FEMALE,JD,priv-house-serv,cross-fit,husband,53100,-43900,26-01-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,WV,Hillsdale,7535 5th Lane,18,4,?,2,3,YES,74360,13520,13520,47320,Toyota,Highlander,2005,Y
204,40,643226,07-04-1992,OH,250/500,1000,1693.83,7000000,447976,MALE,High School,protective-serv,polo,other-relative,44000,-20800,09-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Police,NY,Northbrook,9043 Maple Hwy,6,1,?,1,0,?,53400,5340,5340,42720,Honda,CRV,2003,N
458,59,535879,05-03-2009,IN,100/300,1000,1685.69,0,472236,FEMALE,High School,protective-serv,hiking,wife,31400,0,17-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,VA,Hillsdale,3777 Maple Ave,23,1,?,2,2,YES,71800,14360,14360,43080,Jeep,Grand Cherokee,1995,N
147,31,746630,10-02-1997,IN,250/500,500,1054.92,6000000,468232,FEMALE,PhD,prof-specialty,exercise,own-child,51900,0,16-01-2015,Single Vehicle Collision,Front Collision,Major Damage,Other,NY,Northbrook,5608 Solo St,4,1,?,0,0,?,68240,8530,0,59710,Toyota,Corolla,2013,Y
279,45,598308,28-01-1992,IN,250/500,2000,1333.97,6000000,620819,FEMALE,MD,other-service,bungie-jumping,unmarried,61100,-30700,12-01-2015,Multi-vehicle Collision,Rear Collision,Major Damage,Other,SC,Arlington,6981 Weaver St,21,3,?,1,0,?,61050,11100,11100,38850,Dodge,RAM,2011,Y
308,47,720356,16-09-2013,OH,100/300,1000,1013.61,6000000,452349,FEMALE,Associate,craft-repair,movies,own-child,45700,-41400,03-01-2015,Parked Car,?,Minor Damage,None,NY,Springfield,4369 Maple Lane,7,1,?,1,1,YES,5590,860,860,3870,Suburu,Impreza,2002,N
284,48,724752,16-05-2008,IL,500/1000,500,958.3,0,464646,FEMALE,PhD,machine-op-inspct,exercise,husband,47900,0,22-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,NY,Columbus,6931 Elm St,19,3,?,0,0,?,46860,8520,8520,29820,Volkswagen,Passat,1998,N
108,31,148498,04-01-2002,IN,250/500,2000,1112.04,6000000,472209,FEMALE,PhD,other-service,base-jumping,own-child,52800,-54300,13-01-2015,Parked Car,?,Minor Damage,None,SC,Arlington,7583 Washington Ave,5,1,NO,1,3,NO,4290,780,780,2730,Volkswagen,Passat,1998,N
421,57,110122,02-04-2002,IN,250/500,2000,1206.26,0,459955,FEMALE,Associate,armed-forces,bungie-jumping,own-child,69900,0,31-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NC,Arlington,7552 3rd St,22,3,YES,0,0,NO,78500,15700,7850,54950,Audi,A3,2015,N
266,42,281388,16-07-1998,IL,500/1000,1000,763.67,0,473389,MALE,Associate,prof-specialty,movies,own-child,12800,-49700,04-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Police,SC,Northbrook,1654 Pine St,12,1,YES,1,3,YES,70830,7870,7870,55090,Jeep,Grand Cherokee,2005,N
412,56,728600,15-08-2002,IL,250/500,500,1042.56,0,616767,MALE,High School,handlers-cleaners,yachting,own-child,0,-66100,20-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,SC,Springfield,6058 Andromedia Hwy,19,3,NO,0,2,NO,68040,15120,7560,45360,Suburu,Forrestor,1997,N
31,32,231548,07-09-1999,IL,100/300,2000,1263.48,4000000,442948,FEMALE,JD,other-service,hiking,wife,46800,-87300,07-02-2015,Single Vehicle Collision,Side Collision,Major Damage,Fire,WV,Hillsdale,6536 MLK Hwy,10,1,?,2,0,?,63600,5300,10600,47700,Audi,A5,1997,Y
465,63,531160,12-01-2012,IL,250/500,500,1006.99,6000000,458936,FEMALE,Masters,sales,board-games,own-child,0,0,05-02-2015,Single Vehicle Collision,Side Collision,Minor Damage,Other,WV,Columbus,8198 Embaracadero Lane,7,1,NO,0,3,?,43560,4840,4840,33880,Suburu,Legacy,2015,N
126,31,889003,18-08-1996,OH,250/500,1000,1328.26,0,613921,MALE,Masters,sales,exercise,not-in-family,42300,-45800,02-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Police,WV,Hillsdale,3447 Solo Ave,17,1,NO,1,1,NO,60840,13520,6760,40560,Suburu,Forrestor,2011,N
407,55,193213,11-03-1996,OH,100/300,1000,1250.08,5000000,474598,FEMALE,PhD,tech-support,bungie-jumping,wife,0,-57700,08-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,WV,Arlington,1806 Weaver Ridge,0,3,?,2,3,YES,68160,11360,11360,45440,Ford,Escape,2010,N
101,27,557218,23-11-1997,IL,500/1000,500,982.7,6000000,440865,FEMALE,College,transport-moving,video-games,unmarried,30800,-43700,13-01-2015,Parked Car,?,Minor Damage,None,SC,Arlington,7930 Texas Ave,9,1,NO,1,0,NO,5170,940,470,3760,Toyota,Camry,2001,N
187,37,125591,08-08-2013,IN,500/1000,1000,1412.06,5000000,450947,FEMALE,Masters,protective-serv,reading,not-in-family,60100,0,16-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,NC,Riverwood,7082 Oak Ridge,21,1,?,0,3,?,57700,5770,5770,46160,Nissan,Maxima,2000,N
252,46,227244,30-11-1996,IN,500/1000,2000,1066.7,0,473370,FEMALE,JD,handlers-cleaners,sleeping,own-child,0,0,30-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Other,VA,Northbend,6357 Texas Lane,22,3,NO,0,2,NO,89520,14920,14920,59680,Audi,A3,2014,N
229,43,791425,18-06-1997,IN,250/500,2000,1585.54,0,463153,MALE,High School,protective-serv,reading,not-in-family,42600,-44400,26-02-2015,Vehicle Theft,?,Minor Damage,None,WV,Hillsdale,9322 Rock Hwy,3,1,NO,1,0,YES,4620,420,840,3360,Volkswagen,Jetta,2012,N
246,39,354455,19-04-2007,IN,250/500,1000,1416.08,0,612546,FEMALE,JD,craft-repair,yachting,other-relative,0,-36600,27-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,WV,Northbrook,6684 Solo Lane,16,1,YES,0,3,?,45180,5020,5020,35140,Honda,CRV,2004,N
190,38,601042,19-09-2007,OH,250/500,500,1246.03,0,442919,MALE,JD,craft-repair,movies,unmarried,61900,-50000,28-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Other,NY,Riverwood,4885 Oak Lane,14,1,YES,0,0,NO,45100,9020,4510,31570,Nissan,Maxima,2013,N
95,32,433663,21-12-1996,IN,500/1000,2000,1356.64,0,449352,MALE,Masters,machine-op-inspct,golf,not-in-family,67800,-48600,23-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Police,SC,Springfield,7846 Andromedia Drive,21,3,YES,0,3,YES,83160,15120,15120,52920,Toyota,Camry,2003,N
205,42,471938,03-02-2008,IL,100/300,2000,1387.7,4000000,470104,FEMALE,High School,priv-house-serv,skydiving,other-relative,0,0,18-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Police,WV,Columbus,3915 Embaracadero St,19,1,NO,2,1,NO,86130,15660,15660,54810,Chevrolet,Silverado,1995,N
41,25,564654,16-07-2003,OH,100/300,1000,1004.14,0,459889,MALE,Masters,priv-house-serv,sleeping,wife,35400,0,15-02-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,NC,Hillsdale,4242 Rock Lane,13,3,NO,1,3,NO,48000,9600,4800,33600,Dodge,RAM,1995,N
137,35,645723,05-05-1991,OH,500/1000,500,1107.07,0,478868,FEMALE,High School,protective-serv,movies,husband,0,-45300,04-02-2015,Vehicle Theft,?,Minor Damage,Police,VA,Hillsdale,7405 Oak St,21,1,NO,0,0,YES,3300,600,600,2100,Saab,92x,2008,N
194,34,573572,16-06-1991,IL,100/300,500,1429.96,0,463307,FEMALE,JD,protective-serv,board-games,husband,67800,0,12-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Ambulance,NY,Northbend,9633 4th St,11,3,NO,1,2,YES,57200,11440,5720,40040,Toyota,Camry,2005,N
128,35,437960,03-04-2001,IN,250/500,1000,1074.99,0,453620,FEMALE,Associate,adm-clerical,bungie-jumping,husband,0,-48800,02-01-2015,Vehicle Theft,?,Trivial Damage,Police,VA,Columbus,3492 Britain St,16,1,?,2,0,?,7590,1380,690,5520,Accura,MDX,2012,N
150,37,649800,16-03-2014,OH,500/1000,1000,1007,0,466238,FEMALE,PhD,transport-moving,board-games,unmarried,30400,-89400,27-01-2015,Single Vehicle Collision,Rear Collision,Total Loss,Police,VA,Arlington,7973 4th St,9,1,NO,0,2,?,80080,12320,12320,55440,Chevrolet,Silverado,2013,N
104,30,544225,03-08-2010,OH,100/300,500,1052.85,0,607697,FEMALE,MD,protective-serv,skydiving,other-relative,0,-70100,09-02-2015,Vehicle Theft,?,Minor Damage,Police,WV,Riverwood,3952 Andromedia Lane,8,1,NO,0,0,YES,4800,960,480,3360,BMW,3 Series,2006,N
163,37,390256,25-11-2009,IN,500/1000,1000,1200.33,4000000,477631,FEMALE,High School,craft-repair,cross-fit,own-child,0,-36400,06-02-2015,Vehicle Theft,?,Minor Damage,Police,WV,Springfield,6702 Andromedia St,7,1,?,2,1,YES,3900,390,780,2730,Volkswagen,Jetta,2008,Y
80,26,488597,08-05-2001,IL,100/300,1000,1343,0,443625,MALE,Masters,handlers-cleaners,camping,other-relative,64600,0,03-01-2015,Single Vehicle Collision,Front Collision,Minor Damage,Other,SC,Arlington,5455 Oak Hwy,12,1,?,0,0,NO,90400,9040,9040,72320,BMW,3 Series,1995,N
65,29,133889,14-06-2004,OH,250/500,2000,1441.6,5000000,472223,FEMALE,MD,sales,kayaking,own-child,0,0,12-01-2015,Multi-vehicle Collision,Rear Collision,Minor Damage,Other,NC,Columbus,2253 Maple Ave,21,3,YES,0,0,?,62900,6290,12580,44030,Jeep,Grand Cherokee,1998,N
179,32,931901,07-08-1994,OH,100/300,1000,1433.42,6000000,608328,FEMALE,Associate,protective-serv,base-jumping,own-child,53800,0,22-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Police,NC,Arlington,7897 Lincoln St,4,1,YES,1,2,NO,54200,5420,10840,37940,Nissan,Ultima,2014,Y
372,50,769475,26-08-2004,OH,500/1000,2000,1368.57,0,474860,FEMALE,MD,tech-support,paintball,other-relative,0,0,03-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,NY,Northbend,8811 Maple Hwy,18,3,NO,2,2,YES,51800,5180,10360,36260,Accura,MDX,2003,N
398,55,844062,25-05-1990,OH,250/500,500,862.19,0,606858,MALE,High School,adm-clerical,movies,unmarried,69400,0,23-02-2015,Vehicle Theft,?,Trivial Damage,Police,SC,Northbend,8167 Apache Ave,7,1,?,2,3,?,6600,600,1200,4800,Accura,MDX,2012,N
213,35,844129,20-09-1990,OH,250/500,500,871.46,0,477938,MALE,MD,tech-support,movies,husband,58500,-77700,22-01-2015,Single Vehicle Collision,Side Collision,Total Loss,Fire,SC,Northbrook,5475 Rock Lane,13,1,YES,2,0,YES,74140,13480,6740,53920,Ford,Escape,2007,N
79,25,732169,05-11-2000,OH,500/1000,500,1863.04,0,462698,FEMALE,Associate,priv-house-serv,paintball,not-in-family,53400,-35200,13-02-2015,Single Vehicle Collision,Front Collision,Total Loss,Fire,VA,Northbend,8215 Flute Drive,0,1,NO,2,1,?,67800,13560,6780,47460,Mercedes,C300,1995,N
232,44,221854,03-10-1994,OH,250/500,2000,1181.64,0,454552,MALE,College,other-service,exercise,wife,25800,0,08-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,NY,Northbend,1320 Flute Lane,22,1,?,1,1,YES,55400,5540,11080,38780,Jeep,Grand Cherokee,2002,Y
230,37,776950,11-04-2005,IL,500/1000,1000,1060.74,0,471585,MALE,PhD,tech-support,reading,own-child,0,-51500,09-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Ambulance,SC,Columbus,1229 5th Ave,15,1,YES,2,3,?,49100,9820,4910,34370,Suburu,Impreza,1996,Y
234,41,291006,16-05-1990,IN,100/300,500,951.56,0,455426,FEMALE,JD,transport-moving,video-games,wife,59400,-78600,08-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Police,SC,Riverwood,3884 Pine Lane,3,3,NO,2,1,?,98280,15120,7560,75600,Chevrolet,Tahoe,2007,Y
240,40,845751,11-09-2004,IN,100/300,500,1533.71,9000000,469856,FEMALE,JD,protective-serv,polo,own-child,0,-70900,10-01-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Fire,VA,Northbend,7108 Tree St,18,2,YES,0,2,?,66550,6050,12100,48400,Ford,Escape,2008,N
143,33,889764,30-11-1993,OH,500/1000,1000,1200.09,0,454191,FEMALE,Associate,craft-repair,board-games,unmarried,38400,-5700,26-01-2015,Multi-vehicle Collision,Side Collision,Major Damage,Fire,WV,Arlington,8014 Embaracadero Drive,17,3,?,2,2,?,70400,14080,7040,49280,Accura,RSX,2002,N
266,42,929306,06-03-2003,IN,100/300,500,1093.83,4000000,468454,MALE,Associate,adm-clerical,board-games,other-relative,0,-49600,21-02-2015,Multi-vehicle Collision,Side Collision,Major Damage,Ambulance,WV,Springfield,4937 Flute Drive,18,3,?,1,1,NO,53280,4440,8880,39960,Suburu,Impreza,2015,Y
89,32,515457,18-12-1996,IN,250/500,1000,988.93,0,614187,FEMALE,High School,craft-repair,golf,unmarried,27600,0,23-01-2015,Single Vehicle Collision,Front Collision,Total Loss,Other,NY,Columbus,2889 Francis St,11,1,?,2,3,YES,84590,15380,15380,53830,Dodge,Neon,1999,N
229,37,556270,21-02-1995,IN,500/1000,1000,1331.94,0,433974,FEMALE,Masters,farming-fishing,base-jumping,not-in-family,0,-55400,05-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,NY,Columbus,7504 Flute Drive,17,1,NO,0,2,YES,54560,9920,9920,34720,Saab,95,2004,N
245,40,908935,11-12-2009,IL,500/1000,1000,1361.45,0,604833,MALE,PhD,handlers-cleaners,camping,unmarried,39300,0,15-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Ambulance,OH,Northbend,7570 Cherokee Drive,12,1,YES,0,2,YES,82170,7470,7470,67230,Suburu,Forrestor,1999,N
50,44,525862,18-10-2000,OH,250/500,2000,1188.51,0,447469,MALE,College,handlers-cleaners,bungie-jumping,unmarried,0,-65800,08-01-2015,Multi-vehicle Collision,Front Collision,Total Loss,Police,NY,Northbend,4710 Lincoln Hwy,15,3,?,1,2,NO,61100,6110,12220,42770,Dodge,Neon,2008,N
230,43,490514,09-02-2007,IN,500/1000,2000,1101.83,0,451529,MALE,High School,exec-managerial,cross-fit,other-relative,28900,0,01-01-2015,Multi-vehicle Collision,Front Collision,Minor Damage,Police,NY,Arlington,7511 1st Ave,0,3,?,0,3,YES,51900,5190,10380,36330,BMW,M5,2011,Y
17,39,774895,28-10-2006,IL,250/500,1000,840.95,0,431202,FEMALE,JD,adm-clerical,hiking,unmarried,32500,-80800,26-02-2015,Parked Car,?,Trivial Damage,Police,SC,Arlington,7042 Maple Ridge,9,1,?,2,1,?,3440,430,430,2580,Suburu,Legacy,2002,N
163,36,974522,27-01-2000,IN,250/500,1000,1503.21,0,448190,MALE,MD,other-service,cross-fit,husband,55700,-49900,28-02-2015,Single Vehicle Collision,Side Collision,Total Loss,Ambulance,WV,Springfield,4475 Lincoln Ridge,1,1,YES,2,1,NO,51390,5710,11420,34260,Toyota,Corolla,2013,N
29,32,669809,05-04-2002,OH,100/300,1000,1722.5,0,453713,MALE,High School,other-service,base-jumping,wife,0,-21500,13-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,WV,Columbus,9439 MLK St,22,1,?,0,2,?,76900,7690,7690,61520,Jeep,Wrangler,1995,N
232,42,182953,30-04-2013,IN,100/300,500,944.03,0,440153,MALE,College,handlers-cleaners,kayaking,not-in-family,0,-58400,19-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Other,WV,Riverwood,8269 Sky Hwy,11,1,YES,2,3,?,77000,15400,7700,53900,Toyota,Highlander,2015,Y
235,39,836349,01-05-2013,IL,500/1000,2000,1453.61,4000000,619570,MALE,JD,craft-repair,yachting,other-relative,0,0,13-01-2015,Single Vehicle Collision,Side Collision,Major Damage,Other,NC,Hillsdale,5663 Oak Lane,10,1,?,0,3,?,60320,9280,9280,41760,Chevrolet,Tahoe,2012,Y
295,46,591269,09-01-1999,IN,100/300,500,1672.88,0,478947,FEMALE,High School,armed-forces,dancing,wife,0,0,17-02-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Columbus,4633 5th Lane,5,1,YES,1,1,NO,60700,12140,6070,42490,Honda,Civic,1997,N
22,21,550127,04-07-2007,IN,250/500,1000,1248.05,0,443550,FEMALE,High School,exec-managerial,movies,husband,37500,-54000,15-02-2015,Multi-vehicle Collision,Rear Collision,Total Loss,Police,SC,Arlington,9682 Cherokee Ridge,3,3,YES,1,2,?,53280,5920,0,47360,Chevrolet,Malibu,2015,N
286,43,663190,05-02-1994,IL,100/300,500,1564.43,3000000,477644,FEMALE,MD,prof-specialty,movies,unmarried,77500,-32800,31-01-2015,Single Vehicle Collision,Rear Collision,Minor Damage,Fire,NY,Northbrook,4755 1st St,18,1,?,2,2,YES,34290,3810,3810,26670,Jeep,Grand Cherokee,2013,N
257,44,109392,12-07-2006,OH,100/300,1000,1280.88,0,433981,MALE,MD,other-service,basketball,other-relative,59400,-32200,06-02-2015,Single Vehicle Collision,Rear Collision,Total Loss,Other,WV,Riverwood,5312 Francis Ridge,21,1,NO,0,1,NO,46980,0,5220,41760,Accura,TL,2002,N
94,26,215278,24-10-2007,IN,100/300,500,722.66,0,433696,MALE,MD,exec-managerial,camping,husband,50300,0,23-01-2015,Multi-vehicle Collision,Front Collision,Major Damage,Fire,OH,Springfield,1705 Weaver St,6,3,YES,1,2,YES,36700,3670,7340,25690,Nissan,Pathfinder,2010,N
124,28,674570,08-12-2001,OH,250/500,1000,1235.14,0,443567,MALE,MD,exec-managerial,camping,husband,0,-32100,17-02-2015,Multi-vehicle Collision,Side Collision,Total Loss,Other,OH,Hillsdale,1643 Washington Hwy,20,3,?,0,1,?,60200,6020,6020,48160,Volkswagen,Passat,2012,N
141,30,681486,24-03-2007,IN,500/1000,1000,1347.04,0,430665,MALE,High School,sales,bungie-jumping,own-child,0,-82100,22-01-2015,Parked Car,?,Minor Damage,None,SC,Northbend,6516 Solo Drive,6,1,?,1,2,YES,6480,540,1080,4860,Honda,Civic,1996,N
3,38,941851,16-07-1991,OH,500/1000,1000,1310.8,0,431289,FEMALE,Masters,craft-repair,paintball,unmarried,0,0,22-02-2015,Single Vehicle Collision,Front Collision,Minor Damage,Fire,NC,Northbrook,6045 Andromedia St,20,1,YES,0,1,?,87200,17440,8720,61040,Honda,Accord,2006,N
285,41,186934,05-01-2014,IL,100/300,1000,1436.79,0,608177,FEMALE,PhD,prof-specialty,sleeping,wife,70900,0,24-01-2015,Single Vehicle Collision,Rear Collision,Major Damage,Fire,SC,Northbend,3092 Texas Drive,23,1,YES,2,3,?,108480,18080,18080,72320,Volkswagen,Passat,2015,N
130,34,918516,17-02-2003,OH,250/500,500,1383.49,3000000,442797,FEMALE,Masters,armed-forces,bungie-jumping,other-relative,35100,0,23-01-2015,Multi-vehicle Collision,Side Collision,Minor Damage,Police,NC,Arlington,7629 5th St,4,3,?,2,3,YES,67500,7500,7500,52500,Suburu,Impreza,1996,N
458,62,533940,18-11-2011,IL,500/1000,2000,1356.92,5000000,441714,MALE,Associate,handlers-cleaners,base-jumping,wife,0,0,26-02-2015,Single Vehicle Collision,Rear Collision,Major Damage,Other,NY,Arlington,6128 Elm Lane,2,1,?,0,1,YES,46980,5220,5220,36540,Audi,A5,1998,N
456,60,556080,11-11-1996,OH,250/500,1000,766.19,0,612260,FEMALE,Associate,sales,kayaking,husband,0,0,26-02-2015,Parked Car,?,Minor Damage,Police,WV,Columbus,1416 Cherokee Ridge,6,1,?,0,3,?,5060,460,920,3680,Mercedes,E400,2007,N
'''


# In[2]:


import re
pattern = re.compile(r'([A-Z]{2},[A-Za-z]+),(\d{4})')


# In[3]:


modified=pattern.sub(r'\1 \2', text)


print(modified)


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[5]:


df=pd.read_csv('untitled3.txt')


# In[6]:


pd.set_option('display.max_columns',100,'display.max_rows',200)


# In[7]:


df.shape


# In[8]:


df


# This data has 1000 rows and 38 columns

# we can drop some of the columns which are not contributing to predict the fraud like auto_make,auto_model,policy_number,insured_zip,incident_location

# In[9]:


df.drop(columns=['policy_number','insured_zip','incident_location','auto_model','auto_make'],inplace=True)


# In[10]:


df = df.replace('?', np.nan)


# In[11]:


df['collision_type'].value_counts()


# In[12]:


df.isna().sum()


# to fill the null values in collision_type column with other collision

# In[13]:


df['collision_type']=df['collision_type'].fillna('Other Collision')


# In[14]:


df['property_damage']=df['property_damage'].fillna('Unknown')
df['police_report_available']=df['police_report_available'].fillna('Unknown')


# In[16]:


df.isna().sum()


# In[17]:


df.info()


# In[18]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[22]:


df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])


# In[23]:


df.info()


# In[25]:


categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']


# In[37]:


numerical_features=[feature for feature in df.columns if df[feature].dtypes in ['int64','float64']]


# In[39]:


df.describe()


# In[42]:


plt.figure(figsize=(20,10))
p=1
for i in numerical_features:
    if(p<15):
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
        p+=1
plt.show()


# In[45]:


sns.histplot(data=df,x='months_as_customer',bins=20,hue='fraud_reported')


# In[51]:


sns.barplot(data=df,x='insured_occupation',y='policy_annual_premium', hue='fraud_reported')
plt.xticks(rotation=45,ha='right')


# In[56]:


sns.barplot(data=df,x='policy_state', y='months_as_customer')
plt.xticks(rotation=45,ha='right')


# In[58]:


sns.barplot(data=df,x='policy_state', y='months_as_customer',hue='fraud_reported')
plt.xticks(rotation=45,ha='right')


# In[76]:


sns.countplot(data=df,x='insured_hobbies',hue='fraud_reported')
plt.xticks(rotation=45)


# from the above graph most of the paintball and cross fit as hobbies people are high tend to be fraud

# In[77]:


sns.countplot(data=df,x='incident_type',hue='fraud_reported')
plt.xticks(rotation=45)


# In[78]:


sns.countplot(data=df,x='collision_type',hue='fraud_reported')
plt.xticks(rotation=45)


# In[79]:


sns.countplot(data=df,x='authorities_contacted',hue='fraud_reported')
plt.xticks(rotation=45)


# In[80]:


sns.countplot(data=df,x='incident_severity',hue='fraud_reported')
plt.xticks(rotation=45)


# most of the major damage are fraudulent

# In[81]:


sns.countplot(data=df,x='authorities_contacted',hue='fraud_reported')
plt.xticks(rotation=45)


# In[82]:


sns.histplot(data=df,x='age',hue='fraud_reported')


# In[83]:


plt.figure(figsize=(20,10))
p=1
for i in numerical_features:
    if(p<15):
        plt.subplot(5,4,p)
        sns.distplot(df[i])
        p+=1
plt.show()


# In[84]:


sns.countplot(data=df,x='bodily_injuries',hue='fraud_reported')


# In[86]:


sns.histplot(data=df,x='total_claim_amount',hue='fraud_reported')


# In[94]:


sns.boxplot(data=df,x='fraud_reported',y='capital_gains')


# In[97]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE


# In[88]:


Le=LabelEncoder()


# In[89]:


for i in categorical_features:
    df[i]=Le.fit_transform(df[i])


# In[106]:


x=df.drop(columns=['fraud_reported','policy_bind_date','incident_date'])
y=df['fraud_reported']


# In[107]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[108]:


smt=SMOTE(random_state=0)
X_res,y_res=smt.fit_resample(X_train,y_train)


# In[109]:


model=[
    DecisionTreeClassifier(),
    LogisticRegression(),
    AdaBoostClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier()
    
]


# In[110]:


for i in model:
    i.fit(X_res,y_res)
    y_pred=i.predict(X_test)
    print(i)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    
    scores = cross_val_score(i, X_res, y_res, cv=5)
    print(scores.mean())
    print('diff bet score and accuracy',accuracy_score(y_test,y_pred)-(scores.mean()))
    print('\n')


# In[111]:


params={
    'n_estimators':[100,20,40,60,80],
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[None,100,50]
}


# In[112]:


grid_search=GridSearchCV(RandomForestClassifier(),param_grid=params,cv=10)


# In[113]:


grid_search.fit(X_res,y_res)


# In[114]:


grid_search.best_params_


# In[115]:


rf=RandomForestClassifier(criterion='log_loss',max_depth=100,max_features='log2',n_estimators=60)


# In[116]:


rf.fit(x,y)


# In[ ]:





# In[105]:


df

