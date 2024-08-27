edges_table = {"N69" : ["N70"] ,
               "N70" : ["N69" , "N71"],
               "N71" : ["N70" , "N72"],
               "N72" : ["N71" , "N73"], 
               "N73" : ["N72" , "N74" , "N60"],
               "N74" : ["N73" , "N60" , "N94"],
               "N60" : ["N73" , "N74" , "N45" , "N46" , "N47" , "N61" , "N75"],
               "N41" : ["N42"] ,
               "N42" : ["N41" , "N43"],
               "N43" : ["N42" , "N44"],
               "N44" : ["N43" , "N45"], 
               "N45" : ["N44" , "N46" , "N60"],
               "N46" : ["N45" , "N60" , "N33"],
               "N87" : ["N86"] ,
               "N86" : ["N87" , "N85"],
               "N85" : ["N86" , "N84"],
               "N84" : ["N85" , "N83"], 
               "N83" : ["N84" , "N82" , "N68"],
               "N82" : ["N83" , "N68" , "N94"],
               "N68" : ["N83" , "N82" , "N55" , "N54" , "N53" , "N67" , "N81"],
               "N59" : ["N58"] ,
               "N58" : ["N59" , "N57"],
               "N57" : ["N58" , "N56"],
               "N56" : ["N57" , "N55"], 
               "N55" : ["N56" , "N54" , "N68"],
               "N54" : ["N55" , "N68" , "N34"],
               "N33" : ["N46", "N61", "N47", "N48"],
               "N34" : ["N52", "N53", "N54", "N67"],
               "N94" : ["N74", "N61", "N75", "N76"],
               "N95" : ["N80", "N81", "N82", "N67"],
               "N67" : ["N68", "N95", "N81", "N53", "N34", "N80", "N66", "N52"],
               "N53" : ["N52", "N67", "N34", "N68", "N66"],
               "N81" : ["N80", "N68", "N95", "N66", "N67"],
               "N80" : ["N79", "N81", "N95", "N67", "N66", "N65"],
               "N52" : ["N51", "N34", "N53", "N65", "N66", "N67"],
               "N79" : ["N78", "N80", "N64", "N65", "N66"],
               "N51" : ["N50", "N52", "N64", "N65", "N66"],
               "N65" : ["N64", "N78", "N79", "N80", "N66", "N52", "N51", "N50"],
               "N66" : ["N65", "N79", "N80", "N81", "N67", "N53", "N52", "N51"],
               "N78" : ["N77", "N63", "N64", "N65", "N79"],
               "N64" : ["N63", "N77", "N78", "N79", "N65", "N49", "N50", "N51"],
               "N50" : ["N49", "N63", "N64", "N65", "N51"],
               "N61" : ["N33", "N60", "N94", "N75", "N76", "N62", "N48", "N47"],
               "N75" : ["N61", "N60", "N94", "N76", "N62"],
               "N47" : ["N33", "N60", "N61", "N62", "N48"],
               "N62" : ["N61", "N75", "N76", "N77", "N63", "N49", "N48", "N47"],
               "N76" : ["N62", "N61", "N75", "N94", "N77", "N63"],
               "N48" : ["N33", "N47", "N61", "N62", "N63", "N49"],
               "N63" : ["N62", "N76", "N77", "N78", "N64", "N50", "N49", "N48"],
               "N77" : ["N76", "N78", "N64", "N63", "N62"],
               "N49" : ["N48", "N62", "N63", "N64", "N50"],
                }


nodes_groups = { "g1": ["N41", "N42", "N43", "N44", "N45", "N46", "N69", "N70", "N71", "N72", "N73", "N74", "N54", "N55", "N56", "N57", "N58", "N59", "N82", "N83", "N84", "N85", "N86", "N87"],
                 "g2": ["N33", "N34", "N94", "N95"],
                 "g3": ["N60"],
                 "g4": ["N64", "N78" , "N50"],
                 "g5": ['N75', 'N76', 'N77', 'N61', 'N62', 'N63', 'N47', 'N48', 'N49', 'N51', 'N52', 'N53', 'N65', 'N66', 'N67', 'N79', 'N80', 'N81'],
                 "g6": ["N68"]
                }


def convert_to_state_gcn(edges_table, nodes_groups, groups_states):

    node_names = list(edges_table.keys())
    nodes_features = dict() 

    for group in nodes_groups.keys():
        for node in nodes_groups[group]:
            nodes_features[node] = groups_states[group]

    nodes_state = []
    for node in node_names:
        nodes_state.append(nodes_features[node])

    state_gcn = np.array(nodes_state, dtype=np.float32)
    return state_gcn