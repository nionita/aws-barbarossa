import re
import yaml

'''
The new Config class looks for the file extension and pasrses
either yaml files or old (ini-style) config files
'''

class Config:
    def __init__(self, source):
        old_type = None
        if type(source) != dict:
            # Called with an open file
            if source.name.endswith('.txt'):
                source = OldConfig.readConfig(source)
                old_type = True
            elif source.name.endswith('.yaml') or source.name.endswith('.yml'):
                source = yaml.full_load(source)
                old_type = False
            else:
                raise Exception('Config: cannot determine file type for {}'.format(source.name))

        # Here source is a dictionary
        assert type(source) == dict

        if old_type is None:
            old_type = 'eval' not in source

        if old_type:
            self.__dict__ = source
        else:
            # Some adjustments we need for new config types:
            self.__dict__ = dot_helper(source)
            self.optdir = self.optimization.optdir
            self.save   = self.optimization.save

        # We must now if we have new or old type config
        self.old_type = old_type

    def check(self, key, vtype=None, required=False):
        object = self
        while '.' in key:
            prop, key = key.split('.', 1)
            if not hasattr(object, prop):
                return False
            object = object.__dict__[prop]
        if key in object.__dict__:
            return vtype is None or type(object.__dict__[key]) == vtype
        else:
            return not required

'''
This is a helper class to permit dictionaly access in dot notation
for data which is a combination of dict and list (i.e. dict of dicts,
dict of lists, list of dicts, and so on).
'''
class Dot:
    def __init__(self, data):
        self.__dict__ = data

    def __getattr__(self, name):
        return None

    def __repr__(self):
        rep = 'Dot: {'
        for key, val in self.__dict__.items():
            rep += ' {}: {}'.format(key, repr(val))
        rep += ' }'
        return rep

'''
Transform recursive dictionaries at whatever level (including in lists)
in Dot objects, which can access the keys in dot notation
'''
def dot_recursive(data):
    if type(data) == dict:
        new_dict = dot_helper(data)
        return Dot(new_dict)
    elif type(data) == list:
        return list(map(dot_recursive, data))
    else:
        return data

'''
Transform dict values recursive in Dot elements, where necessary
'''
def dot_helper(data):
    new_dict = {}
    for name, val in data.items():
        new_dict[name] = dot_recursive(val)
    return new_dict

# This is the old config file, with *.ini syntax
# It is still supported, yet not further developed
# It is applied if config file has a suffix of .txt

'''
A config class for tuning configuration
It represents a config file with the following structure:

# This line is a comment
# The file begins with section 0, which defines match parameters
# (the selfplay program, the current directory for the selfplay execution, input pgn file
# for the games, search depth, number of games per match) and optimization hyper parameters
name: test
optdir: C:/Learn/dspsa
selfplay: C:/astra/SelfPlay-dnp.exe
playdir: C:/Learn/dspsa
ipgnfile: C:/astra/open-moves/open-moves.fen
depth: 4
games: 8
laststep: 10
msteps: 10

#alpha: 0.501
#rend: 300

[params]
# Section params defines the parameters to be optimized (can be empty or not appear at all)
# with param name, starting value and scale (name + 2 values)
# For bayesian optimization it should be: param name, minimum, starting and maximum value
# (name + 3 values)
epMovingMid:  156, 3
epMovingEnd:  156, 3
epMaterMinor: 1, 1

[weights]
# Section weights defines the weights to be optimized (can be empty or not appear at all)
# with param name, starting mid game value, starting end game value, and scale (name + 3 values)
# For bayesian optimization it should be: param name, mid minimum, mid starting and mid maximum value,
# end minimum, end starting and end maximum value (name + 6 values)
kingSafe: 1, 0, 1
kingOpen: 2, 4, 1
kingPlaceCent: 8, 1, 1
'''

class OldConfig:
    # These are acceptable fields in section 0, with their type and maybe default value
    # S is string, I integer and F float
    fields = {
        'name': 'S',
        'method': ('S', 'Bayes'),
        'regressor': ('S', 'GP'),
        'isotropic': ('I', 0),      # Isotropic kernel (when GP, default: false)
        'acq_func': ('S', 'EI'),    # Could be LCB, EI, gp_hedge and some others
        'normalize': ('S', ''),     # normalization for Bayes GP: '', 'X', 'Y' or 'XY'
        'elo': ('I', 0),            # Play result in Elo difference (default: elowish)
        'prop_scale': ('F', 0),     # Parameter changes are proportional with this sigmoid scale
                                    # (default: 0, which means linear)
        'probto': ('F', 0.01),      # Accepted probability of timeout in Play
        'nu': ('F', 1.5),           # Nu param for the Matern kernel when GP
        'fix_noise': ('I', 1),      # Fix the noise from number of games
        'simul': ('F', 0.0),        # Simulation with rosenbrock: noise variance - 0: no simulation
        'in_real': ('I', 0),        # Optimize in real parameters (default: integer)
        'triang': ('S', 'DF'),
        'optdir': ('S', '.'),
        'selfplay': 'S',
        'playdir': ('S', '.'),
        'ipgnfile': 'S',
        'ipgnlen': 'I',
        'depth': ('I', 4),
        'nodes': 'I',
        'games': ('I', 16),
        'laststep': ('F', 0.1),
        'alpha': ('F', 0.501),
        'beta': 'F',
        'msteps': ('I', 1000),  # number of optimizing steps
        'isteps': ('I', 0),     # number of initial steps for GP
        'ropoints': ('I', 5),   # number of restart optimizer points for GP
        'rend': 'I',
        'save': ('I', 10),
        'parallel': ('I', 1),   # number of parallel playing games
        'play_chunk': 'I'       # number of games played in a chunk
    }

    mandatory_fields = ['name', 'selfplay', 'ipgnfile', 'ipgnlen']

    '''
    A config can be initialized either with a file name or with a dictionary
    When called with a file name, the file will be read and transformed into a dictionary which
    will be used for the config creation
    When called with a dictionary, the dictionary keys will be used as attributes of the
    new created config object
    '''
    def __init__(self, source):
        if type(source) != dict:
            # Called with an open file
            source = OldConfig.readConfig(source)
        # Here source is a dictionary
        for name, val in source.items():
            self.__setattr__(name, val)

    @staticmethod
    def accept_data_type(field_name, field_type):
        if field_type not in ['S', 'I', 'F']:
            raise Exception('OldConfig: wrong field type {} for field {}'.format(field_type, field_name))

    @staticmethod
    def create_defaults():
        values = dict()
        for field_name, field_spec in OldConfig.fields.items():
            if type(field_spec) == str:
                OldConfig.accept_data_type(field_name, field_spec)
                values[field_name] = None
            else:
                field_type, field_ini = field_spec
                OldConfig.accept_data_type(field_name, field_type)
                values[field_name] = field_ini
        return values

    # Reading the configuration file
    # There are 3 sections: 0, 1 and 2
    # Section 0 contains general parameter (like optimization method)
    # We have a list with possible field names in this section, and some may have a default
    # Section 1 contains params, i.e. one value per param
    # Section 2 contains weights, i.e. 2 values per weight
    # Number of config values per param/weight depends on optimization method:
    # When using DSPSA, we have the starting params/weights (1/2) plus scale (1)
    # When using Bayes, we have the starting params/weights (1/2) plus inf/sup limit (2/4)
    # All params/weights must have the same number of values
    # So we denote variant 1 as the one with 2/3 values, and variant 2 as the one with 3/6 values

    @staticmethod
    def readConfig(cof):
        # Transform the config file to a dictionary
        values = OldConfig.create_defaults()
        seen = set()
        sectionNames = [dict(), dict()]
        section = 0
        lineno = 0
        error = False
        expect = None
        for line in cof:
            lineno += 1
            # split the comment path
            line = re.split('#', line)[0].lstrip().rstrip()
            if len(line) > 0:
                if line == '[params]':
                    section = 1
                    if expect == 3:
                        expect = 2
                    elif expect == 6:
                        expect = 3
                    else:
                        lenghts = [2, 3]
                elif line == '[weights]':
                    section = 2
                    if expect == 2:
                        expect = 3
                    elif expect == 3:
                        expect = 6
                    else:
                        lenghts = [3, 6]
                else:
                    parts = re.split(r':\s*', line, 1)
                    name = parts[0]
                    val = parts[1]
                    if section == 0:
                        if name in OldConfig.fields:
                            field_type = OldConfig.fields[name]
                            if type(field_type) == tuple:
                                field_type = field_type[0]
                            if field_type == 'S':
                                values[name] = val
                            elif field_type == 'I':
                                values[name] = int(val)
                            elif field_type == 'F':
                                values[name] = float(val)
                            else:
                                raise Exception('Cannot be here!')
                        else:
                            print('OldConfig error in line {:d}: unknown config name {:s}'.format(lineno, name))
                            error = True
                    else:
                        vals = re.split(r',\s*', val)
                        if expect is None and len(vals) in lenghts or expect is not None and len(vals) == expect:
                            if expect is None:
                                expect = len(vals)

                            if name in seen:
                                print('OldConfig error in line {:d}: name {:s} already seen'.format(lineno, name))
                                error = True
                            else:
                                seen.add(name)
                                sectionNames[section-1][name] = [int(v) for v in vals]
                        else:
                            if expect is None:
                                print('OldConfig error in line {:d}: should have {:d} or {:d} values, it has {:d}'.format(lineno, section+1, *lenghts, len(vals)))
                            else:
                                print('OldConfig error in line {:d}: should have {:d} values, it has {:d}'.format(lineno, expect, len(vals)))
                            error = True
        for mand in OldConfig.mandatory_fields:
            if values[mand] is None:
                print('OldConfig does not define mandatory field "{}"'.format(mand))
                error = True

        if expect is None:
            print('OldConfig does not have parameters or weights')
            error = True

        if error:
            raise Exception('OldConfig file has errors')

        hasScale = False

        # Collect the eval parameters
        variant = None
        values['pnames'] = []
        values['pinits'] = []
        values['pscale'] = []
        values['pmin'] = []
        values['pmax'] = []
        for name, vals in sectionNames[0].items():
            assert variant is None or len(vals) == 2 and variant == 1 or len(vals) == 3 and variant == 2
            if len(vals) == 2:
                if variant is None:
                    variant = 1
                # We have: name: init, scale
                values['pnames'].append(name)
                values['pinits'].append(vals[0])
                values['pscale'].append(vals[1])
                if values['pscale'] != 1:
                    hasScale = True
            else:
                if variant is None:
                    variant = 2
                # We have: name: init, min, max
                values['pnames'].append(name)
                values['pinits'].append(vals[0])
                values['pmin'].append(vals[1])
                values['pmax'].append(vals[2])

        # Collect the eval weights
        for name, vals in sectionNames[1].items():
            assert variant is None or len(vals) == 3 and variant == 1 or len(vals) == 6 and variant == 2
            if len(vals) == 3:
                if variant is None:
                    variant = 1
                # We have: name: initMid, initEnd, scale
                values['pnames'].append('mid.' + name)
                values['pinits'].append(vals[0])
                values['pscale'].append(vals[2])
                values['pnames'].append('end.' + name)
                values['pinits'].append(vals[1])
                values['pscale'].append(vals[2])
                if values['pscale'] != 1:
                    hasScale = True
            else:
                if variant is None:
                    variant = 2
                # We have: name: initMid, minMid, maxMid, initEnd, minEnd, maxEnd
                values['pnames'].append('mid.' + name)
                values['pinits'].append(vals[0])
                values['pmin'].append(vals[1])
                values['pmax'].append(vals[2])
                values['pnames'].append('end.' + name)
                values['pinits'].append(vals[3])
                values['pmin'].append(vals[4])
                values['pmax'].append(vals[5])

        if variant == 1:
            values['pmin'] = None
            values['pmax'] = None
            if not hasScale:
                values['pscale'] = None
        else:
            values['pscale'] = None

        return values

# vim: tabstop=4 shiftwidth=4 expandtab
