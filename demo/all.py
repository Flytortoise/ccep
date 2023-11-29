import numpy as np

base_ea_acc = [93.67, 93.65, 93.51, 93.56, 93.75, 93.69, 93.57, 93.58, 93.6, 93.27, 93.26, 93.52, 93.41, 93.3, 93.24, 92.91, 92.93, 92.91, 92.87, 92.85]
lstpa_ea1_acc = [93.31, 92.79, 92.56, 92.4, 91.72, 90.89, 90.57, 90.05, 89.89, 89.38, 89.11, 88.19, 87.52, 87.08, 86.27, 85.79, 84.91, 84.45, 84.72, 84.01]
lstpa_ea2_acc = [92.87, 92.48, 92.08, 91.59, 91.06, 90.62, 89.85, 89.51, 89.0, 88.55, 87.92, 87.56, 87.0, 85.89, 85.67, 85.34, 85.18, 84.65, 84.22, 83.91]
print("base_ea_acc:",np.mean(base_ea_acc))
print("lstpa_ea1_acc:",np.mean(lstpa_ea1_acc))
print("lstpa_ea2_acc:",np.mean(lstpa_ea2_acc))

base_ea_float = [17.504608146534572, 30.04365688935231, 38.66590973291093, 46.25788719640447, 52.201346603952935, 58.58916299876409, 62.67227049351395, 67.38899049388895, 70.59886211797833, 73.51517075822251, 76.34974645003554, 78.54376225917797, 80.36580551461236, 81.57042388222384, 83.05667979475507, 84.19219252391915, 85.12840189641229, 85.81690938072705, 86.66865602315936, 87.45390579652252]
lstpa_ea1_float = [50.37556088541633, 63.9530667103092, 74.34816941837487, 81.55311785645573, 87.10756236385485, 89.0173637040286, 90.4425632035525, 92.8586899942281, 94.55969934340536, 96.03123507090962, 96.99044441405408, 97.75562908231646, 98.11997913799738, 98.26105904215771, 98.39586414262794, 98.65701622409982, 98.72239595901623, 98.87042960928787, 98.88044305464068, 98.97058571763985]
lstpa_ea2_float = [56.05475373828499, 72.6302209016233, 77.54892181395887, 81.95794587333447, 87.9878424213619, 90.96660612147734, 92.96610556385289, 95.1894700285008, 96.44741148934644, 97.07395942711425, 97.70668026351832, 98.09678936918326, 98.46227057096179, 98.69950681011093, 98.7898672951627, 98.84168219360004, 98.8931518886675, 98.98760513544404, 98.99983374190467, 99.01799067487234]
print("base_ea_float:",np.mean(base_ea_float))
print("lstpa_ea1_float:",np.mean(lstpa_ea1_float))
print("lstpa_ea2_float:",np.mean(lstpa_ea2_float))

base_ea_params = [18.987682050681222, 34.05715691817609, 45.4908357654352, 55.02586050101543, 62.54107862422321, 69.03075730921617, 74.03848963838534, 78.40200799027363, 81.85179175327644, 84.72446313055445, 87.10238954258692, 88.98405641631196, 90.56805297029128, 91.67552614066368, 92.61971232185918, 93.60116901020716, 94.34537149808357, 94.93895069577619, 95.53754733374424, 95.99823108541779]
lstpa_ea1_params = [56.72036751148707, 70.19003288158135, 76.37009146553426, 80.15218056486503, 85.17974245852706, 86.99129861095636, 87.52244003458297, 91.88112109365252, 94.21414408407095, 96.49334301770476, 97.69239114523208, 98.4578310166148, 98.99352283155505, 99.1860604299973, 99.34392965121718, 99.51956674936991, 99.58379265374685, 99.63188535255725, 99.66173645334494, 99.6879112115904]
lstpa_ea2_params = [57.75310617584179, 67.71323887646167, 70.0149629143108, 73.21117912381881, 81.23481340259714, 85.61656668038012, 89.53447361780529, 92.98364354502972, 94.37846525309183, 95.20556225956153, 96.86658853159939, 98.18893758504461, 98.84648914624917, 99.21395659727342, 99.35892859501931, 99.47332223002267, 99.56599141617384, 99.62679451887352, 99.63948490637871, 99.67480048001957]
print("base_ea_params:",np.mean(base_ea_params))
print("lstpa_ea1_params:",np.mean(lstpa_ea1_params))
print("lstpa_ea2_params:",np.mean(lstpa_ea2_params))
