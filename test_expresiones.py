import unittest

import numpy as np

import expresiones as ex


def cara(x, prob, t):
    p = np.full(7, (1 - prob) / 6)
    p[3] = prob
    return ex.Cara((x, 0, 100, 100), p, t)


class TestExpresiones(unittest.TestCase):
    def test_alineacion_recupera_puntos(self):
        ang = 0.4
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        puntos = 3 * ex.PUNTOS_ESTANDAR @ rot.T + [200, 80]
        m = ex.matriz_alineacion(puntos)
        np.testing.assert_allclose(puntos @ m[:, :2].T + m[:, 2], ex.PUNTOS_ESTANDAR, atol=1e-6)

    def test_softmax(self):
        p = ex.softmax(np.array([1000.0, 0.0]))
        self.assertAlmostEqual(p.sum(), 1)
        self.assertGreater(p[0], 0.99)

    def test_etiqueta(self):
        self.assertEqual(cara(0, 0.9, 0).etiqueta(0.5), "feliz 90%")
        self.assertEqual(cara(0, 0.3, 0).etiqueta(0.5), "no seguro")

    def test_suaviza_misma_cara(self):
        r = ex.suavizar([cara(0, 0.9, 0)], [cara(10, 0.1, 0.2)], 1.0)[0]
        self.assertGreater(r.probabilidades[3], 0.5)
        self.assertAlmostEqual(r.probabilidades.sum(), 1)

    def test_no_mezcla_caras_lejanas(self):
        r = ex.suavizar([cara(0, 0.9, 0)], [cara(500, 0.1, 0.2)], 1.0)[0]
        self.assertAlmostEqual(r.probabilidades[3], 0.1)
        self.assertEqual(r.id, 0)

    def test_conserva_identidad_y_tiempo(self):
        previa = cara(0, 0.9, 0)
        previa.id, previa.nombre, previa.desde = 4, "se ve feliz", 2.0
        r = ex.suavizar([previa], [cara(10, 0.9, 5)], 1.0)[0]
        self.assertEqual((r.id, r.nombre, r.desde), (4, "se ve feliz", 2.0))

    def test_frase(self):
        self.assertEqual(cara(0, 0.9, 0).frase(0.5), "se ve feliz")
        self.assertEqual(cara(0, 0.3, 0).frase(0.5), "- no seguro")

    def test_orientacion(self):
        self.assertEqual(ex.orientacion(ex.PUNTOS_ESTANDAR), "de frente")
        lado = ex.PUNTOS_ESTANDAR.copy()
        lado[2, 0] += 25
        self.assertEqual(ex.orientacion(lado), "de lado")


if __name__ == "__main__":
    unittest.main()
