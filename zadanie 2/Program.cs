using System;
using System.Collections.Generic;

public class NeuronSiec
{
    static Random random = new Random();
    static double beta = 1.0;
    static double parametrUczenia = 0.3;
    static int liczbaEpok = 50000;

    public static void Main()
    {
        
        int[] architektura = new int[] { 2, 2, 2, 2 };
        var siec = SiecNeuronowa(architektura);

        List<(double[] wejscie, double[] oczekiwane)> dane = new List<(double[], double[])>
        {
            (new double[] {0, 0}, new double[] {0, 1}),
            (new double[] {0, 1}, new double[] {1, 0}),
            (new double[] {1, 0}, new double[] {1, 0}),
            (new double[] {1, 1}, new double[] {0, 0})
        };

        for (int epoka = 0; epoka < liczbaEpok; epoka++)
        {
            MieszajDane(dane);

            foreach (var (wejscie, oczekiwane) in dane)
            {
                List<double[]> aktywacje = new List<double[]> { wejscie };

                for (int l = 0; l < siec.Count; l++)
                {
                    double[] aktualna = new double[siec[l].Count];
                    for (int i = 0; i < siec[l].Count; i++)
                    {
                        aktualna[i] = Sigmoid(Sumator(siec[l][i], aktywacje[l]), beta);
                    }
                    aktywacje.Add(aktualna);
                }

                double[][] delty = new double[siec.Count][];
                for (int l = siec.Count - 1; l >= 0; l--)
                {
                    delty[l] = new double[siec[l].Count];
                    for (int i = 0; i < siec[l].Count; i++)
                    {
                        if (l == siec.Count - 1)
                        {
                            double blad = Blad(oczekiwane[i], aktywacje[l + 1][i]);
                            delty[l][i] = Pochodna(parametrUczenia, blad, aktywacje[l + 1][i]);
                        }
                        else
                        {
                            double suma = 0;
                            for (int k = 0; k < siec[l + 1].Count; k++)
                            {
                                suma += delty[l + 1][k] * siec[l + 1][k][i];
                            }
                            delty[l][i] = Pochodna(parametrUczenia, suma, aktywacje[l + 1][i]);
                        }

                        for (int j = 0; j < siec[l][i].Length - 1; j++)
                        {
                            siec[l][i][j] += aktywacje[l][j] * delty[l][i];
                        }
                        siec[l][i][siec[l][i].Length - 1] += delty[l][i]; 
                    }
                }
            }

            if (epoka % 10000 == 0)
            {
                Console.WriteLine($"Epoka {epoka}/{liczbaEpok}");
                foreach (var (wejscie, oczekiwane) in dane)
                {
                    double[] wyjscie = PrzekazPrzezSiec(siec, wejscie);
                    double blad1 = Math.Abs(oczekiwane[0] - wyjscie[0]);
                    double blad2 = Math.Abs(oczekiwane[1] - wyjscie[1]);

                    Console.WriteLine($"Wejście: [{string.Join(", ", wejscie)}] " +
                                      $"-> Wyjście: [{wyjscie[0]:F4}, {wyjscie[1]:F4}] " +
                                      $"(Oczekiwane: [{oczekiwane[0]}, {oczekiwane[1]}], " +
                                      $"Błąd: [{blad1:F4}, {blad2:F4}])");
                }
            }

        }
        Console.WriteLine("Wyniki po treningu:");
        foreach (var (wejscie, oczekiwane) in dane)
        {
            double[] wyjscie = PrzekazPrzezSiec(siec, wejscie);
            double blad1 = Math.Abs(oczekiwane[0] - wyjscie[0]);
            double blad2 = Math.Abs(oczekiwane[1] - wyjscie[1]);

            Console.WriteLine($"Wejście: [{string.Join(", ", wejscie)}] " +
                              $"-> Wyjście: [{wyjscie[0]:F4}, {wyjscie[1]:F4}] " +
                              $"(Oczekiwane: [{oczekiwane[0]}, {oczekiwane[1]}], " +
                              $"Błąd: [{blad1:F4}, {blad2:F4}])");
        }


    }

    static double[] PrzekazPrzezSiec(List<List<double[]>> siec, double[] wejscie)
    {
        double[] wejscieAktualne = wejscie;
        foreach (var warstwa in siec)
        {
            double[] wyjscie = new double[warstwa.Count];
            for (int i = 0; i < warstwa.Count; i++)
            {
                wyjscie[i] = Sigmoid(Sumator(warstwa[i], wejscieAktualne), beta);
            }
            wejscieAktualne = wyjscie;
        }
        return wejscieAktualne;
    }

    static void MieszajDane(List<(double[] wejscie, double[] oczekiwane)> dane)
    {
        int n = dane.Count;
        for (int i = 0; i < n; i++)
        {
            int j = random.Next(i, n);
            var temp = dane[i];
            dane[i] = dane[j];
            dane[j] = temp;
        }
    }

    static double Sigmoid(double x, double beta)
    {
        return 1.0 / (1.0 + Math.Exp(-beta * x));
    }

    static double Sumator(double[] wagi, double[] wejscia)
    {
        double wynik = 0;
        double bias = wagi[wagi.Length - 1];
        for (int i = 0; i < wejscia.Length; i++) wynik += wejscia[i] * wagi[i];
        return wynik + bias;
    }

    static double Pochodna(double ParametrUczenia, double Blad, double wyjscie)
    {
        return Blad * ParametrUczenia * wyjscie * (1 - wyjscie);
    }

    static double Blad(double wyjscieOczekiwane, double WyjscieObliczone)
    {
        return wyjscieOczekiwane - WyjscieObliczone;
    }

    static List<List<double[]>> SiecNeuronowa(int[] architektura)
    {
        List<List<double[]>> siec = new List<List<double[]>>();
        for (int i = 1; i < architektura.Length; i++)
        {
            List<double[]> warstwa = new List<double[]>();
            for (int j = 0; j < architektura[i]; j++)
            {
                double[] wagi = new double[architektura[i - 1] + 1];
                for (int k = 0; k < wagi.Length; k++)
                {
                    wagi[k] = random.NextDouble() * 10 - 5;
                }
                warstwa.Add(wagi);
            }
            siec.Add(warstwa);
        }
        return siec;
    }
}
