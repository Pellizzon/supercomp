import os
import os.path as osp
import io
import sys
import pprint
from grading_tools import TestConfiguration, ProgramTest, CheckOutputMixin, CheckStderrMixin, CheckMultiCorePerformance
from colorama import Fore


def compila_programa(ext, nome, flags, nvcc=False):
    compilador = 'g++'
    if nvcc:
        compilador = 'nvcc -arch=sm_50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70'
    arquivos = ' '.join([arq for arq in os.listdir('.') if arq.split('.')[-1] == ext])
    ret = os.system(f'{compilador} {arquivos} -O3 {flags} -o {nome} 2>/dev/null  > /dev/null')
    if ret != 0 :
        raise IOError(f'Erro de compilação em {nome}!')

def green(s):
    return Fore.GREEN + str(s) + Fore.RESET

def red(s):
    return Fore.RED + str(s) + Fore.RESET

def test_result(b):
    if b:
        return green(b)
    else:
        return red(b)

class BaseMMS:
    def parse_input(self, inp):
        lines = inp.split('\n')
        N, M = [int(i) for i in lines[0].split()]
        values = [int(i) for i in lines[1].split()]
        return N, M, values
    
    def parse_output(self, out, M):
        lines = out.split('\n')
        if len(lines) < M+1:
            print(f'Menos de {M+1} linhas detectadas na saída. É necessário uma linha por pessoa. Linhas vazias não contam.')
            return -1, []
            
        mms = int(lines[0])
        objects_person = [[] for _ in range(M)]
        
        for i in range(1, M+1):
            objects_person[i-1].extend([int(o) for o in lines[i].split()])

        return mms, objects_person
    
    def calc_mms(self, values, objects_person):
        min_mms = sum(values)
        person = None
        for i in range(len(objects_person)):
            i_mms = sum([values[o] for o in objects_person[i]])
            if i_mms < min_mms:
                min_mms = i_mms
                person = i
        return min_mms, person

    def objects_person_from_aloc(self, aloc, M):
        objects_person = [[] for _ in range(M)]
        for i, o in enumerate(aloc):
            objects_person[o].append(i)
        
        return objects_person
    
    def test_solucao_final_valida(self, test, stdout, stderr):
        N, M, values = self.parse_input(test.input)
        try:
            mms_out, objects_person = self.parse_output(stdout, M)
            return mms_out == self.calc_mms(values, objects_person)[0]
        except ValueError:
            print('Format error')
            return False


class TesteBuscaLocal(ProgramTest, BaseMMS):
    def test_roda_iter_vezes(self, test, stdout, stderr):
        if test.environ['DEBUG'] == '0': return True

        N, M, values = self.parse_input(test.input)
        return len(stderr.split('\n')) == int(test.environ['ITER'])

    def test_solucao_final_eh_melhore_e_valida(self, test, stdout, stderr):
        if test.environ['DEBUG'] == '0': return True

        N, M, values = self.parse_input(test.input)
        max_val = 0
        try:
            for sol in stderr.split('\n'):
                val, *aloc = [int(i) for i in sol.split()]

                if val > max_val:
                    max_val = val
                
            mms_out, objects_person_out = self.parse_output(stdout, M)
            mms_out_recalc, _ = self.calc_mms(values, objects_person_out)
        except ValueError:
            print('Format error')
            return False

        return max_val == mms_out and mms_out == mms_out_recalc
        
    def test_toda_solucao_valida(self, test, stdout, stderr):
        if test.environ['DEBUG'] == '0': return True

        N, M, values = self.parse_input(test.input)
        try:
            for sol in stderr.split('\n'):
                val, *aloc = [int(i) for i in sol.split()]

                objects_person = self.objects_person_from_aloc(aloc, M)
                min_mms, person = self.calc_mms(values, objects_person)
                
                if min_mms != val:
                    print('Erro no cálculo do MMS:', sol)
                    print('Calculado:', min_mms)
                    return False

            return True
        except ValueError:
            print('Format error')
            return False
    
    def test_toda_solucao_sem_troca(self, test, stdout, stderr):
        if test.environ['DEBUG'] == '0': return True

        N, M, values = self.parse_input(test.input)
        try:
            for sol in stderr.split('\n'):
                val, *aloc = [int(i) for i in sol.split()]
                if len(aloc) != N:
                    print(f'Vetor de alocação tem menos que {N} objetos')
                    return False
                sol_objects_person = self.objects_person_from_aloc(aloc, M)
                sol_min_mms, sol_person = self.calc_mms(values, sol_objects_person)
                
                for i in range(N):
                    for j in range(N):
                        if i == j: continue
                        aloc[i], old_aloc_i = aloc[j], aloc[i]
                        objects_person = self.objects_person_from_aloc(aloc, M)
                        swap_min_mms, swap_person = self.calc_mms(values, objects_person)
                        
                        if swap_min_mms > sol_min_mms:
                            print('Doação de', i, 'para', j, 'possível')
                            print('Alocação antiga:', sol_objects_person)
                            print('Alocação nova:', objects_person)
                            print('Novo MMS', swap_min_mms)
                            return False
                        
                        aloc[i], aloc[j] = old_aloc_i, aloc[i]
                        
            return True
        except ValueError:
            print('Format error')
            return False

class TesteMultiCore(TesteBuscaLocal, CheckMultiCorePerformance):
    pass

class TesteExaustivoRapido(ProgramTest, BaseMMS):
    def test_MMS_minimo(self, test, stdout, stderr):
        N, M, values = self.parse_input(test.input)
        mms_out, _ = self.parse_output(stdout, M)
        mms_esperado, _ = self.parse_output(test.output, M)

        return mms_out == mms_esperado


class TesteExaustivo(TesteExaustivoRapido, CheckStderrMixin):
    pass

class TesteHeuristico(ProgramTest, CheckOutputMixin):
    pass


def testa_heuristico():
    os.chdir('heuristico')
    try:
        compila_programa('cpp', 'heuristico', '')
    except IOError:
        res = False
    else:
        tests = TestConfiguration.from_pattern('.', 'in*.txt', 'out*txt')
        tester = TesteHeuristico('./heuristico', tests)
        res = tester.main()

    os.chdir('..')
    return res

def testa_busca_exaustiva():
    os.chdir('busca-global')
    try:
        compila_programa('cpp', 'global', '')
    except IOError:
        res = False
    else:
        testes_basicos = {
            f'in{i}.txt': TestConfiguration.from_file(f'in{i}.txt', f'out{i}.txt', f'err{i}.txt', 
                                                      environ={'DEBUG':'1'}, time_limit=3000)    
            for i in range(1, 7)
        }
        tester = TesteExaustivo('./global', testes_basicos)
        res = tester.main()

    os.chdir('..')
    return res

def testa_busca_exaustiva_desempenho_1():
    os.chdir('busca-global')
    try:
        compila_programa('cpp', 'global', '')
    except IOError:
        res = False
    else:
        testes_basicos = {
            'in5.txt': TestConfiguration.from_file('in5.txt', 'out5.txt', 'err5.txt', environ={'DEBUG': '0'}, time_limit=3),
            'in6.txt': TestConfiguration.from_file('in6.txt', 'out6.txt', 'err6.txt', environ={'DEBUG': '0'}, time_limit=34),
            'in7.txt': TestConfiguration.from_file('in7.txt', 'out7.txt', 'err7.txt', environ={'DEBUG': '0'},time_limit=3),
            'in9.txt': TestConfiguration.from_file('in9.txt', 'out9.txt', 'err9.txt', environ={'DEBUG': '0'},time_limit=5),
            'in10.txt': TestConfiguration.from_file('in10.txt', 'out10.txt', 'err10.txt', environ={'DEBUG': '0'},time_limit=28),
        }

        tester = TesteExaustivoRapido('./global', testes_basicos)
        res = tester.main()

    os.chdir('..')
    return res

def testa_busca_exaustiva_desempenho_2():
    os.chdir('busca-global')
    try:
        compila_programa('cpp', 'global', '')
    except IOError:
        res = False
    else:
        testes_basicos = {
            'in5.txt': TestConfiguration.from_file('in5.txt', 'out5.txt', 'err5.txt', environ={'DEBUG': '0'}, time_limit=2.5),
            'in6.txt': TestConfiguration.from_file('in6.txt', 'out6.txt', 'err6.txt', environ={'DEBUG': '0'}, time_limit=20),
            'in7.txt': TestConfiguration.from_file('in7.txt', 'out7.txt', 'err7.txt', environ={'DEBUG': '0'},time_limit=2.5),
            'in9.txt': TestConfiguration.from_file('in9.txt', 'out9.txt', 'err9.txt', environ={'DEBUG': '0'},time_limit=4),
            'in10.txt': TestConfiguration.from_file('in10.txt', 'out10.txt', 'err10.txt', environ={'DEBUG': '0'},time_limit=16),
        }

        tester = TesteExaustivoRapido('./global', testes_basicos)
        res = tester.main()

    os.chdir('..')
    return res


def testa_busca_exaustiva_desempenho_3():
    os.chdir('busca-global')
    try:
        compila_programa('cpp', 'global', '')
    except IOError:
        res = False
    else:
        testes_basicos = {
            'in5.txt': TestConfiguration.from_file('in5.txt', 'out5.txt', 'err5.txt', environ={'DEBUG': '0'}, time_limit=2),
            'in6.txt': TestConfiguration.from_file('in6.txt', 'out6.txt', 'err6.txt', environ={'DEBUG': '0'}, time_limit=8),
            'in7.txt': TestConfiguration.from_file('in7.txt', 'out7.txt', 'err7.txt', environ={'DEBUG': '0'},time_limit=2),
            'in9.txt': TestConfiguration.from_file('in9.txt', 'out9.txt', 'err9.txt', environ={'DEBUG': '0'},time_limit=3),
            'in10.txt': TestConfiguration.from_file('in10.txt', 'out10.txt', 'err10.txt', environ={'DEBUG': '0'},time_limit=8),
        }

        tester = TesteExaustivoRapido('./global', testes_basicos)
        res = tester.main()

    os.chdir('..')
    return res


def testa_busca_local_sequencial():
    os.chdir('busca-local')
    try:
        compila_programa('cpp', 'local', '')
    except IOError as e:
        print(e)
        res = False
    else:
        testes_basicos = {
            inp:TestConfiguration.from_file(inp, out, environ={'DEBUG': '1', 'ITER': '10'}, time_limit=300)
            for inp,out in [('in7.txt', 'out7.txt'),('in8.txt', 'out8.txt'),
                            ('in9.txt', 'out9.txt'),('in10.txt', 'out10.txt'),
                            ('in11.txt', 'out11.txt'),('in12.txt', 'out12.txt')]
        }
        tester = TesteBuscaLocal('./local', testes_basicos)
        res = tester.main()
    
    os.chdir('..')
    return res

def testa_busca_local_omp():
    os.chdir('busca-local')
    try:
        compila_programa('cpp', 'local-omp', '-fopenmp')
    except IOError:
        res = False
    else:
        testes_basicos = {
            inp:TestConfiguration.from_file(inp, out, environ={'DEBUG': '1', 'ITER': '10'}, time_limit=300)
            for inp,out in [('in7.txt', 'out7.txt'),('in8.txt', 'out8.txt'),
                            ('in9.txt', 'out9.txt')]
        }
        teste_sequencial = TesteBuscaLocal('./local-omp', testes_basicos)
        res = teste_sequencial.main()

        testes_grandes = {
            inp:TestConfiguration.from_file(inp, out, environ={'DEBUG': '0', 'ITER': '100000'}, time_limit=15)
            for inp,out in [('in10.txt', 'out10.txt'),
                            ('in11.txt', 'out11.txt'),('in12.txt', 'out12.txt'),
                            ('in13.txt', 'out13.txt'),('in14.txt', 'out14.txt'),
                            ('in15.txt', 'out15.txt')]
        }
        
        teste_multi_core = TesteMultiCore('./local-omp', testes_grandes)
        res = teste_multi_core.main() and res
    
    os.chdir('..')
    return res

def testa_busca_local_desempenho_1():
    os.chdir('busca-local')
    try:
        compila_programa('cpp', 'local-omp', '-fopenmp')
    except IOError:
        res = False
    else:
        test = lambda i,t:  TestConfiguration.from_file(f'in{i}.txt', 
                                f'out{i}.txt', time_limit=t, 
                                environ={'DEBUG': '0', 'ITER': '100000'})
        testes_grandes = {
            'in14.txt': test(14, 4),
            'in15.txt': test(15, 12),
            'in16.txt': test(16, 43),
            'in17.txt': test(17, 180)
        }
        teste_multi_core = TesteMultiCore('./local-omp', testes_grandes)
        res = teste_multi_core.main()

    os.chdir('..')
    return res

def testa_busca_local_desempenho_2():
    os.chdir('busca-local')
    try:
        compila_programa('cpp', 'local-omp', '-fopenmp')
    except IOError:
        res = False
    else:
        test = lambda i,t:  TestConfiguration.from_file(f'in{i}.txt', 
                                f'out{i}.txt', time_limit=t, 
                                environ={'DEBUG': '0', 'ITER': '100000'})
        testes_grandes = {
            'in14.txt': test(14, 2),
            'in15.txt': test(15, 8),
            'in16.txt': test(16, 20),
            'in17.txt': test(17, 60)
        }
        teste_multi_core = TesteMultiCore('./local-omp', testes_grandes)
        res = teste_multi_core.main()

    os.chdir('..')
    return res

def testa_busca_local_desempenho_3():
    os.chdir('busca-local')
    try:
        compila_programa('cpp', 'local-omp', '-fopenmp')
    except IOError:
        res = False
    else:
        test = lambda i,t:  TestConfiguration.from_file(f'in{i}.txt', 
                                f'out{i}.txt', time_limit=t, 
                                environ={'DEBUG': '0', 'ITER': '100000'})
        testes_grandes = {
            'in14.txt': test(14, 1.5),
            'in15.txt': test(15, 5),
            'in16.txt': test(16, 13),
            'in17.txt': test(17, 25)
        }
        teste_multi_core = TesteMultiCore('./local-omp', testes_grandes)
        res = teste_multi_core.main()

    os.chdir('..')
    return res


def testa_busca_local_gpu():
    os.chdir('busca-local')
    try:
        compila_programa('cu', 'local-gpu', '', nvcc=True)
    except IOError as e:
        print(e)
        res = False
    else:
        testes_basicos = {
            inp:TestConfiguration.from_file(inp, out, environ={'DEBUG': '1', 'ITER': '10'}, time_limit=300)
            for inp,out in [('in7.txt', 'out7.txt'),('in8.txt', 'out8.txt'),
                            ('in9.txt', 'out9.txt'),('in10.txt', 'out10.txt'),
                            ('in11.txt', 'out11.txt'),('in12.txt', 'out12.txt')]
        }
        tester = TesteBuscaLocal('./local-gpu', testes_basicos)
        res = tester.main()
    
    os.chdir('..')
    return res

def testa_busca_local_gpu_1():
    os.chdir('busca-local')
    try:
        compila_programa('cu', 'local-gpu', '', nvcc=True)
    except IOError:
        res = False
    else:
        test = lambda i,t:  TestConfiguration.from_file(f'in{i}.txt', 
                                f'out{i}.txt', time_limit=t, 
                                environ={'DEBUG': '0', 'ITER': '100000'})
        testes_grandes = {
            'in14.txt': test(14, 5),
            'in15.txt': test(15, 5),
            'in16.txt': test(16, 5),
            'in17.txt': test(17, 5)
        }
        teste_multi_core = TesteBuscaLocal('./local-gpu', testes_grandes)
        res = teste_multi_core.main()

    os.chdir('..')
    return res


if __name__ == "__main__":

    ignorar = io.StringIO()

    testesD = {
        'heuristico': ('Heuristico (sequencial)', testa_heuristico),
        'local': ('Busca local (sequencial)', testa_busca_local_sequencial),
        'global': ('Busca exaustiva (sequencial)', testa_busca_exaustiva),
        'multi-core': ('Busca local (paralela)', testa_busca_local_omp),
        'global-level1': ('Busca global (desempenho nível 1)', testa_busca_exaustiva_desempenho_1),
        'global-level2': ('Busca global (desempenho nível 2)', testa_busca_exaustiva_desempenho_2),
        'global-level3': ('Busca global (desempenho nível 3)', testa_busca_exaustiva_desempenho_3),

        'local-level1': ('Busca local (desempenho nível 1)', testa_busca_local_desempenho_1),
        'local-level2': ('Busca local (desempenho nível 2)', testa_busca_local_desempenho_2),
        'local-level3': ('Busca local (desempenho nível 3)', testa_busca_local_desempenho_3),

        'local-gpu': ('Busca local (GPU)', testa_busca_local_gpu),
        'local-gpu-level1': ('Busca local (GPU - level 1)', testa_busca_local_gpu_1),
    }

    if len(sys.argv) > 1:
        tst = sys.argv[1]
        if tst in testesD:
            tst = testesD[tst]
            print(tst[0], ':', test_result(tst[1]()))
            sys.exit(0)
        else:
            print('Testes disponíveis:')
            pprint.pprint(testesD)
    else:
        print('Rodando todos os testes')
        print('Testes disponíveis:')
        pprint.pprint(testesD)

        resultados = [osp.split(os.getcwd())[1]] 
        testes = testesD.values()
        for t in testes:
            resultados.append('X' if t[1]() else ' ')

        print(','.join(resultados), file=sys.stderr)




