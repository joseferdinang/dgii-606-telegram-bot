"""
Synthetic Dominican Republic Invoice Generator for ML Training.

Generates realistic invoice OCR representations based on real Dominican company profiles,
layouts, and tax rules across:
1. Combustible (GLP, Gasolina Premium/Regular, Gasoil Óptimo/Regular, GNV) - Ley 112-00
2. Ferreterías (Ferretería Ochoa, Americana, Bellón, Haché, etc.)
3. Equipos Electrónicos y de Cómputo (Cecomsa, Omega Tech, Corripio, Plaza Lama, etc.)
4. Supermercados (Nacional, Sirena, Jumbo, Bravo)
5. Restaurantes (Adrian Tropical, SBG, etc. con 10% propina legal)
6. Farmacias (Carol, Los Hidalgos, GBC con exentos y copago ARS)

Over 175 real company profiles and branches are included.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any, List, Tuple


# =====================================================================
# DIRECTORIO DE EMPRESAS REALES DOMINICANAS (+175 PERFILES Y SUCURSALES)
# =====================================================================

COMBUSTIBLE_STATIONS = [
    # TotalEnergies
    {"name": "TOTALENERGIES CHURCHILL", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Av. Winston Churchill esq. Andres Julio Aybar, D.N.", "phone": "809-262-1000", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES LUPERON", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Av. Luperon No. 45, Herrera, Santo Domingo Oeste", "phone": "809-560-2211", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES 27 DE FEBRERO", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Av. 27 de Febrero esq. Leopoldo Navarro, D.N.", "phone": "809-688-4455", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES SANTIAGO LAS COLINAS", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Av. Imbert No. 120, Las Colinas, Santiago", "phone": "809-575-3344", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES SAN ISIDRO", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Autopista de San Isidro Km 3, Santo Domingo Este", "phone": "809-788-9900", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES BAVARO", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Boulevard Turistico del Este Km 14, Bavaro", "phone": "809-552-0192", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES NUNEZ DE CACERES", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Av. Nunez de Caceres No. 204, El Millon, D.N.", "phone": "809-530-8812", "fuel_type": "combustible"},
    {"name": "TOTALENERGIES AUTOPISTA DUARTE KM 9", "company": "TOTALENERGIES MARKETING DOMINICANA, S.A.", "rnc": "101-06874-4", "address": "Aut. Duarte Km 9, Santo Domingo Oeste", "phone": "809-561-1234", "fuel_type": "combustible"},

    # Shell (Isla Dominicana de Petróleo)
    {"name": "SHELL DUARTE KM 6 1/2", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Autopista Duarte Km 6 1/2, Santo Domingo", "phone": "809-565-1100", "fuel_type": "combustible"},
    {"name": "SHELL LINCOLN", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. Abraham Lincoln esq. Roberto Pastoriza, D.N.", "phone": "809-541-2299", "fuel_type": "combustible"},
    {"name": "SHELL TIRADENTES", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. Tiradentes esq. Salvador Sanquintin, Naco", "phone": "809-567-3388", "fuel_type": "combustible"},
    {"name": "SHELL SAN VICENTE", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. San Vicente de Paul No. 88, Los Mina", "phone": "809-594-5566", "fuel_type": "combustible"},
    {"name": "SHELL LAS AMERICAS KM 10", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Autopista Las Americas Km 10, Santo Domingo Este", "phone": "809-598-1122", "fuel_type": "combustible"},
    {"name": "SHELL SANTIAGO MONUMENTO", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. Francia No. 45, Santiago de los Caballeros", "phone": "809-582-7711", "fuel_type": "combustible"},
    {"name": "SHELL KENNEDY DEFENSE", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. John F. Kennedy esq. Defillo, D.N.", "phone": "809-566-4433", "fuel_type": "combustible"},
    {"name": "SHELL CHARLES DE GAULLE", "company": "ISLA DOMINICANA DE PETROLEO CORPORATION", "rnc": "101-00817-2", "address": "Av. Charles de Gaulle esq. Carretera Mella", "phone": "809-748-0011", "fuel_type": "combustible"},

    # Sunix Petroleum
    {"name": "SUNIX ABRAHAM LINCOLN", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Av. Abraham Lincoln No. 1009, Piantini, D.N.", "phone": "809-683-1100", "fuel_type": "combustible"},
    {"name": "SUNIX 27 DE FEBRERO", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Av. 27 de Febrero No. 412, Bella Vista, D.N.", "phone": "809-535-9988", "fuel_type": "combustible"},
    {"name": "SUNIX SAN ISIDRO", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Av. San Isidro esq. Charles de Gaulle, SDE", "phone": "809-699-2244", "fuel_type": "combustible"},
    {"name": "SUNIX ECOLOGICA", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Av. Ecologica Km 5, Santo Domingo Este", "phone": "809-483-5566", "fuel_type": "combustible"},
    {"name": "SUNIX SANTIAGO DUARTE", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Aut. Duarte Km 2 1/2, El Embrujo, Santiago", "phone": "809-581-8899", "fuel_type": "combustible"},
    {"name": "SUNIX LUPERON HERRERA", "company": "SUNIX PETROLEUM, S.R.L.", "rnc": "130-19273-1", "address": "Av. Luperon No. 101, Zona Industrial Herrera", "phone": "809-560-7788", "fuel_type": "combustible"},

    # Propagas (Especialistas GLP Vehicular e Industrial)
    {"name": "PROPAGAS MAJUTA KM 5", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Av. Jacobo Majluta Km 5 1/2, Santo Domingo Norte", "phone": "809-364-1000", "fuel_type": "glp"},
    {"name": "PROPAGAS SAN MARTIN", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Av. San Martin No. 302, Ensanche La Fe, D.N.", "phone": "809-565-3311", "fuel_type": "glp"},
    {"name": "PROPAGAS PROLONGACION 27", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Prolongacion Av. 27 de Febrero Km 13, Las Caobas", "phone": "809-561-8822", "fuel_type": "glp"},
    {"name": "PROPAGAS CHARLES DE GAULLE", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Av. Charles de Gaulle No. 54, Santo Domingo Este", "phone": "809-596-7744", "fuel_type": "glp"},
    {"name": "PROPAGAS INDEPENDENCIA KM 10", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Av. Independencia Km 10 1/2, Los Kilometros, D.N.", "phone": "809-532-1144", "fuel_type": "glp"},
    {"name": "PROPAGAS SANTIAGO CANABONCITO", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Autopista Duarte Km 5, Canaboncito, Santiago", "phone": "809-583-4422", "fuel_type": "glp"},
    {"name": "PROPAGAS VILLA MELLA", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Av. Hermanas Mirabal No. 182, Villa Mella, SDN", "phone": "809-568-9911", "fuel_type": "glp"},
    {"name": "PROPAGAS CORREDOR LAS AMERICAS", "company": "PROPANO Y DERIVADOS, S.A.", "rnc": "101-03373-8", "address": "Marginal Las Americas Km 14, Santo Domingo Este", "phone": "809-598-6633", "fuel_type": "glp"},

    # Tropigas (Envasadoras GLP Vehicular y Doméstico)
    {"name": "TROPIGAS CENTRAL SAN MARTIN", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Av. San Martin #278, Ensanche La Fe, D.N.", "phone": "809-227-8767", "fuel_type": "glp"},
    {"name": "TROPIGAS CARRETERA SANCHEZ KM 8", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Carr. Sanchez Km 8 1/2, Miramar, D.N.", "phone": "809-533-0099", "fuel_type": "glp"},
    {"name": "TROPIGAS MELLA EL TAMARINDO", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Carretera Mella Km 9 1/2, El Tamarindo, SDE", "phone": "809-695-1122", "fuel_type": "glp"},
    {"name": "TROPIGAS AUTOPISTA SAN ISIDRO", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Av. San Isidro No. 90, Santo Domingo Este", "phone": "809-788-3344", "fuel_type": "glp"},
    {"name": "TROPIGAS SANTIAGO BARTOLOME COLON", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Av. Bartolome Colon No. 65, Santiago", "phone": "809-582-1200", "fuel_type": "glp"},
    {"name": "TROPIGAS DUARTE LOS ALCARRIZOS", "company": "TROPIGAS DOMINICANA, S.R.L.", "rnc": "101-72699-7", "address": "Aut. Duarte Km 14, Los Alcarrizos", "phone": "809-560-4499", "fuel_type": "glp"},

    # Eco Petróleo Dominicana
    {"name": "ECO PETROLEO AV. ESPANA", "company": "ECO PETROLEO DOMINICANA, S.A.", "rnc": "130-84918-2", "address": "Av. Espana No. 34, Ensanche Isabelita, SDE", "phone": "809-766-0044", "fuel_type": "combustible"},
    {"name": "ECO PETROLEO SANTO DOMINGO OESTE", "company": "ECO PETROLEO DOMINICANA, S.A.", "rnc": "130-84918-2", "address": "Av. Beisbolistas No. 89, Manoguayabo, SDO", "phone": "809-561-9012", "fuel_type": "combustible"},
    {"name": "ECO PETROLEO CARRETERA MELLA", "company": "ECO PETROLEO DOMINICANA, S.A.", "rnc": "130-84918-2", "address": "Carretera Mella Km 7, El Almirante, SDE", "phone": "809-695-8833", "fuel_type": "combustible"},

    # Next Combustibles
    {"name": "NEXT COMBUSTIBLES LINCOLN", "company": "NEXT COMBUSTIBLES, S.R.L.", "rnc": "131-20491-8", "address": "Av. Abraham Lincoln esq. Bolivar, D.N.", "phone": "809-535-4400", "fuel_type": "combustible"},
    {"name": "NEXT 30 DE MARZO", "company": "NEXT COMBUSTIBLES, S.R.L.", "rnc": "131-20491-8", "address": "Av. 30 de Marzo esq. Mexico, San Carlos, D.N.", "phone": "809-687-1234", "fuel_type": "combustible"},
    {"name": "NEXT SANTIAGO ESTRELLA SADHALA", "company": "NEXT COMBUSTIBLES, S.R.L.", "rnc": "131-20491-8", "address": "Av. Estrella Sadhala No. 102, Santiago", "phone": "809-583-9900", "fuel_type": "combustible"},

    # Petromóvil
    {"name": "PETROMOVIL KENNEDY", "company": "PETROMOVIL DE COMBUSTIBLES, S.A.", "rnc": "101-54829-1", "address": "Av. John F. Kennedy Km 7, Los Jardines, D.N.", "phone": "809-567-8899", "fuel_type": "combustible"},
    {"name": "PETROMOVIL LAS AMERICAS", "company": "PETROMOVIL DE COMBUSTIBLES, S.A.", "rnc": "101-54829-1", "address": "Aut. Las Americas Km 18, Boca Chica", "phone": "809-523-4411", "fuel_type": "combustible"},
]

FERRETERIA_STORES = [
    # Ferretería Ochoa
    {"name": "FERRETERIA OCHOA LAS COLINAS", "company": "FERRETERIA OCHOA, S.A.", "rnc": "102-00343-2", "address": "Av. Imbert No. 53, Las Colinas, Santiago", "phone": "809-971-8000"},
    {"name": "FERRETERIA OCHOA SANTO DOMINGO", "company": "FERRETERIA OCHOA, S.A.", "rnc": "102-00343-2", "address": "Av. Luperon No. 42, Zona Industrial Herrera, SDO", "phone": "809-227-2200"},
    {"name": "FERRETERIA OCHOA BELLA TERRA MALL", "company": "FERRETERIA OCHOA, S.A.", "rnc": "102-00343-2", "address": "Av. Juan Pablo Duarte, Mall Bella Terra, Santiago", "phone": "809-580-8000"},
    {"name": "FERRETERIA OCHOA CARRETERA LICEY", "company": "FERRETERIA OCHOA, S.A.", "rnc": "102-00343-2", "address": "Carretera Duarte Km 2, Licey al Medio, Santiago", "phone": "809-580-8001"},

    # Ferretería Americana
    {"name": "FERRETERIA AMERICANA KENNEDY", "company": "FERRETERIA AMERICANA, S.A.S.", "rnc": "101-00991-8", "address": "Av. John F. Kennedy esq. Paseo de los Aviadores", "phone": "809-567-9211"},
    {"name": "FERRETERIA AMERICANA SAN MARTIN", "company": "FERRETERIA AMERICANA, S.A.S.", "rnc": "101-00991-8", "address": "Av. San Martin No. 250, Ensanche La Fe, D.N.", "phone": "809-541-1122"},
    {"name": "FERRETERIA AMERICANA GALERIA 360", "company": "FERRETERIA AMERICANA, S.A.S.", "rnc": "101-00991-8", "address": "Centro Comercial Galeria 360, Av. Kennedy, D.N.", "phone": "809-563-3600"},
    {"name": "FERRETERIA AMERICANA LUPERON", "company": "FERRETERIA AMERICANA, S.A.S.", "rnc": "101-00991-8", "address": "Av. Luperon esq. Romulo Betancourt, Bella Vista", "phone": "809-537-8899"},

    # Bellón
    {"name": "BELLON MATRIZ SANTIAGO", "company": "BELLON, S.A.S.", "rnc": "102-00062-1", "address": "Av. Bartolome Colon No. 34, Santiago", "phone": "809-582-4141"},
    {"name": "BELLON PONTEZUELA", "company": "BELLON, S.A.S.", "rnc": "102-00062-1", "address": "Carretera Don Pedro Km 2, Pontezuela, Santiago", "phone": "809-241-4141"},
    {"name": "BELLON SAN FRANCISCO", "company": "BELLON, S.A.S.", "rnc": "102-00062-1", "address": "Av. Antonio Guzman Fernandez No. 80, SFM", "phone": "809-588-4141"},

    # Ferretería Haché
    {"name": "FERRETERIA HACHE KENNEDY", "company": "ANTONIO P. HACHE & CO., C. POR A.", "rnc": "101-00248-2", "address": "Av. John F. Kennedy esq. Tiradentes, D.N.", "phone": "809-566-5111"},
    {"name": "FERRETERIA HACHE SANTIAGO", "company": "ANTONIO P. HACHE & CO., C. POR A.", "rnc": "101-00248-2", "address": "Av. Las Carreras esq. Mella, Santiago", "phone": "809-582-2244"},

    # Ferremix
    {"name": "FERREMIX MATERIALES", "company": "FERREMIX, S.R.L.", "rnc": "130-49182-3", "address": "Av. Monumental No. 18, Santo Domingo Oeste", "phone": "809-561-5588"},
    {"name": "FERRETERIA CUESTA", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. 27 de Febrero esq. A. Lincoln, D.N.", "phone": "809-537-4001"},
    {"name": "CONSTRURAMA FERRETERIA CENTRAL", "company": "CONSTRURAMA DOMINICANA, S.R.L.", "rnc": "131-84920-1", "address": "Carretera Mella Km 8, Santo Domingo Este", "phone": "809-594-7711"},
]

TECH_STORES = [
    # Cecomsa
    {"name": "CECOMSA PIANTINI", "company": "CECOMSA, S.R.L.", "rnc": "102-31616-3", "address": "Av. Gustavo Mejia Ricart No. 71, Piantini, D.N.", "phone": "809-540-0000"},
    {"name": "CECOMSA SANTIAGO", "company": "CECOMSA, S.R.L.", "rnc": "102-31616-3", "address": "Av. Los Jardines Metropolitanos No. 22, Santiago", "phone": "809-580-0000"},
    {"name": "CECOMSA CORAL MALL", "company": "CECOMSA, S.R.L.", "rnc": "102-31616-3", "address": "Coral Mall, Aut. San Isidro, Santo Domingo Este", "phone": "809-788-0000"},

    # Omega Tech
    {"name": "OMEGA TECH CHURCHILL", "company": "OMEGA TECH, S.A.", "rnc": "122-02152-3", "address": "Av. Winston Churchill No. 80, Ens. Piantini, D.N.", "phone": "809-683-4400"},
    {"name": "OMEGA TECH SANTIAGO", "company": "OMEGA TECH, S.A.", "rnc": "122-02152-3", "address": "Av. 27 de Febrero No. 89, Santiago", "phone": "809-582-4400"},
    {"name": "OMEGA TECH SAN VICENTE", "company": "OMEGA TECH, S.A.", "rnc": "122-02152-3", "address": "Av. San Vicente de Paul No. 112, SDE", "phone": "809-594-4400"},

    # Distribuidora Corripio
    {"name": "DISTRIBUIDORA CORRIPIO KENNEDY", "company": "DISTRIBUIDORA CORRIPIO, S.A.S.", "rnc": "101-00369-3", "address": "Av. John F. Kennedy Km 6 1/2, D.N.", "phone": "809-227-3000"},
    {"name": "DISTRIBUIDORA CORRIPIO SAN VICENTE", "company": "DISTRIBUIDORA CORRIPIO, S.A.S.", "rnc": "101-00369-3", "address": "Av. San Vicente de Paul No. 45, SDE", "phone": "809-597-3000"},
    {"name": "DISTRIBUIDORA CORRIPIO SANTIAGO", "company": "DISTRIBUIDORA CORRIPIO, S.A.S.", "rnc": "101-00369-3", "address": "Av. Bartolome Colon No. 10, Santiago", "phone": "809-581-3000"},

    # Plaza Lama
    {"name": "PLAZA LAMA 27 DE FEBRERO", "company": "PLAZA LAMA, S.A.", "rnc": "101-17111-1", "address": "Av. 27 de Febrero esq. Winston Churchill, D.N.", "phone": "809-686-2171"},
    {"name": "PLAZA LAMA DUARTE", "company": "PLAZA LAMA, S.A.", "rnc": "101-17111-1", "address": "Av. Duarte No. 94, Santo Domingo", "phone": "809-687-2171"},
    {"name": "PLAZA LAMA HERRERA", "company": "PLAZA LAMA, S.A.", "rnc": "101-17111-1", "address": "Av. Isabel Aguiar No. 120, Herrera, SDO", "phone": "809-560-2171"},

    # GBM Dominicana
    {"name": "GBM DOMINICANA", "company": "GBM DOMINICANA, S.A.", "rnc": "101-02489-1", "address": "Av. Winston Churchill No. 1099, Torre Acropolis 18, D.N.", "phone": "809-566-4141"},
    {"name": "COMPUMARKET RD", "company": "COMPUMARKET, S.R.L.", "rnc": "130-98214-2", "address": "Av. Romulo Betancourt No. 1412, Bella Vista, D.N.", "phone": "809-532-8877"},
]

SUPERMARKETS = [
    # Centro Cuesta Nacional
    {"name": "SUPERMERCADOS NACIONAL 27 DE FEBRERO", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. 27 de Febrero esq. Abraham Lincoln, D.N.", "phone": "809-537-4001"},
    {"name": "SUPERMERCADOS NACIONAL LOPE DE VEGA", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. Lope de Vega No. 59, Naco, D.N.", "phone": "809-541-4001"},
    {"name": "SUPERMERCADOS NACIONAL SANTIAGO", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. Juan Pablo Duarte No. 140, Santiago", "phone": "809-583-4001"},
    {"name": "JUMBO LUPERON", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. Luperon esq. Gustavo Mejia Ricart, D.N.", "phone": "809-563-4001"},
    {"name": "JUMBO MEGACENTRO", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. San Vicente de Paul, Megacentro, SDE", "phone": "809-591-4001"},
    {"name": "JUMBO AGORA MALL", "company": "CENTRO CUESTA NACIONAL, S.A.S.", "rnc": "101-01992-1", "address": "Av. John F. Kennedy esq. A. Lincoln, Agora Mall", "phone": "809-563-4002"},

    # Grupo Ramos
    {"name": "LA SIRENA CHURCHILL", "company": "GRUPO RAMOS, S.A.", "rnc": "101-79682-2", "address": "Av. Winston Churchill No. 1010, Ens. Paraiso, D.N.", "phone": "809-472-4444"},
    {"name": "LA SIRENA LUPERON", "company": "GRUPO RAMOS, S.A.", "rnc": "101-79682-2", "address": "Av. Luperon esq. Prolongacion Gustavo M. Ricart", "phone": "809-560-4444"},
    {"name": "LA SIRENA AUTOPISTA DUARTE", "company": "GRUPO RAMOS, S.A.", "rnc": "101-79682-2", "address": "Aut. Duarte Km 10 1/2, Santo Domingo Oeste", "phone": "809-564-4444"},
    {"name": "SUPER POLA SARASOTA", "company": "GRUPO RAMOS, S.A.", "rnc": "101-79682-2", "address": "Av. Sarasota No. 34, Bella Vista, D.N.", "phone": "809-535-4444"},

    # Supermercados Bravo
    {"name": "SUPERMERCADOS BRAVO CHURCHILL", "company": "BRAVO, S.A.", "rnc": "101-60246-5", "address": "Av. Winston Churchill esq. Angel Severo Cabral", "phone": "809-563-2000"},
    {"name": "SUPERMERCADOS BRAVO ENRIQUILLO", "company": "BRAVO, S.A.", "rnc": "101-60246-5", "address": "Av. Enriquillo No. 67, Los Cacicazgos, D.N.", "phone": "809-534-2000"},
    {"name": "SUPERMERCADOS BRAVO SAN VICENTE", "company": "BRAVO, S.A.", "rnc": "101-60246-5", "address": "Av. San Vicente de Paul No. 18, SDE", "phone": "809-598-2000"},
    {"name": "SUPERMERCADOS BRAVO SANTIAGO", "company": "BRAVO, S.A.", "rnc": "101-60246-5", "address": "Av. Juan Pablo Duarte No. 80, Santiago", "phone": "809-581-2000"},
]

RESTAURANTS = [
    {"name": "ADRIAN TROPICAL MALECON", "company": "GASTRONOMICA DEL CARIBE, S.R.L.", "rnc": "1-30-49182-3", "address": "Av. George Washington #1, Malecon, D.N.", "phone": "809-682-1234"},
    {"name": "ADRIAN TROPICAL LINCOLN", "company": "GASTRONOMICA DEL CARIBE, S.R.L.", "rnc": "1-30-49182-3", "address": "Av. Abraham Lincoln No. 803, Piantini, D.N.", "phone": "809-565-9234"},
    {"name": "ADRIAN TROPICAL 27 DE FEBRERO", "company": "GASTRONOMICA DEL CARIBE, S.R.L.", "rnc": "1-30-49182-3", "address": "Av. 27 de Febrero esq. Nuñez de Caceres", "phone": "809-534-7234"},
    {"name": "SBG CAFE BLUE MALL", "company": "SBG DOMINICANA, S.R.L.", "rnc": "1-30-19284-1", "address": "Blue Mall 4to Nivel, Av. Winston Churchill", "phone": "809-955-3200"},
    {"name": "CHEF PEPPER TIRADENTES", "company": "CHEF PEPPER, S.R.L.", "rnc": "1-30-09481-2", "address": "Av. Tiradentes esq. Roberto Pastoriza, Naco", "phone": "809-565-4067"},
    {"name": "LA LOCANDA PIANTINI", "company": "LA LOCANDA, S.R.L.", "rnc": "1-31-29401-2", "address": "Calle Manuel de Jesus Troncoso No. 14, Piantini", "phone": "809-567-8890"},
]

PHARMACIES = [
    {"name": "FARMACIA CAROL 27 DE FEBRERO", "company": "GRUPO CAROL, S.A.S.", "rnc": "101-87186-5", "address": "Av. 27 de Febrero No. 242, Bella Vista, D.N.", "phone": "809-563-2222"},
    {"name": "FARMACIA CAROL CHURCHILL", "company": "GRUPO CAROL, S.A.S.", "rnc": "101-87186-5", "address": "Av. Winston Churchill esq. Victor Garrido Puello", "phone": "809-540-2222"},
    {"name": "FARMACIA LOS HIDALGOS 27 DE FEBRERO", "company": "FARMACIA LOS HIDALGOS, S.A.", "rnc": "101-05308-9", "address": "Av. 27 de Febrero No. 241, La Esperilla, D.N.", "phone": "809-541-4141"},
    {"name": "FARMACIAS MEDICAR GBC INDEPENDENCIA", "company": "FARMACIA MEDICAR GBC, S.R.L.", "rnc": "130-06442-3", "address": "Av. Independencia Km 8 1/2, Santo Domingo", "phone": "809-532-4400"},
]


# =====================================================================
# PRODUCTOS REALES POR SECTOR
# =====================================================================

FUEL_PRODUCTS = [
    # Combustibles tradicionales
    {"desc": "GASOLINA PREMIUM", "price": 290.10, "unit": "GL"},
    {"desc": "GASOLINA REGULAR", "price": 272.50, "unit": "GL"},
    {"desc": "GASOIL OPTIMO", "price": 239.10, "unit": "GL"},
    {"desc": "GASOIL REGULAR", "price": 221.60, "unit": "GL"},
    {"desc": "DIESEL OPTIMO ULTRA BAJO AZUFRE", "price": 239.10, "unit": "GL"},
]

GLP_PRODUCTS = [
    {"desc": "GLP VEHICULAR", "price": 132.60, "unit": "GL"},
    {"desc": "GLP AUTOMOTRIZ", "price": 132.60, "unit": "GL"},
    {"desc": "GAS LICUADO DE PETROLEO GLP", "price": 132.60, "unit": "GL"},
]

HARDWARE_PRODUCTS = [
    {"desc": "CEMENTO TITAN GRIS TIPO PORTLAND 42.5KG", "price": 485.00, "unit": "BLS"},
    {"desc": "CEMENTO CIBAO ESTRUCTURAL 42.5KG", "price": 475.00, "unit": "BLS"},
    {"desc": "VARILLA CORRUGADA 3/8 X 20 GRADO 60", "price": 395.00, "unit": "ATD"},
    {"desc": "VARILLA ACERO 1/2 X 20 METALDOM", "price": 620.00, "unit": "ATD"},
    {"desc": "PINTURA TROPICAL ACRILICA PLUS BLANCO", "price": 5200.00, "unit": "CUBA"},
    {"desc": "PINTURA POPULAR SUPER ACRILICA BLANCO", "price": 1350.00, "unit": "GL"},
    {"desc": "TUBO PVC SANITARIO 3 X 19 SDR-41", "price": 640.00, "unit": "TUBO"},
    {"desc": "TUBO PVC PRESION 1/2 X 19 SCH-40", "price": 280.00, "unit": "TUBO"},
    {"desc": "CLAVOS DE ACERO 2 1/2 (50 LBS)", "price": 1850.00, "unit": "CAJA"},
    {"desc": "PEGA SOLDADURA PVC TANGIT 1/4 GL", "price": 450.00, "unit": "UND"},
    {"desc": "CERRADURA YALE POMO ENTRADA PRINCIPAL", "price": 1250.00, "unit": "PZA"},
    {"desc": "DISCO CORTE METAL 4 1/2 DEWALT", "price": 95.00, "unit": "PZA"},
    {"desc": "ALUZINC CALIBRE 26 AZUL 12 PIES", "price": 890.00, "unit": "PL"},
]

TECH_PRODUCTS = [
    {"desc": "LAPTOP DELL LATITUDE 3440 I5 16GB 512SSD", "price": 54900.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "LAPTOP LENOVO THINKPAD E14 RYZEN 5 16GB", "price": 48500.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "MONITOR LG 24 IPS FHD HDMI 75HZ", "price": 8900.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "MONITOR SAMSUNG 27 CURVO ESSENTIAL", "price": 11500.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "TECLADO LOGITECH INALAMBRICO MK270", "price": 1450.00, "has_serial": False, "warranty": "30 DIAS"},
    {"desc": "MOUSE INALAMBRICO LOGITECH M185", "price": 650.00, "has_serial": False, "warranty": "30 DIAS"},
    {"desc": "DISCO SOLIDO SSD KINGSTON NV2 1TB M.2", "price": 4200.00, "has_serial": True, "warranty": "24 MESES"},
    {"desc": "MEMORIA RAM CORSAIR DDR4 16GB 3200MHZ", "price": 2800.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "IMPRESORA MULTIFUNCIONAL EPSON ECOTANK L3250", "price": 14900.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "UPS FORZA NT-501 500VA 250W 6 TOMAS", "price": 2650.00, "has_serial": True, "warranty": "12 MESES"},
    {"desc": "SMART TV SAMSUNG 55 CRYSTAL 4K UHD", "price": 28900.00, "has_serial": True, "warranty": "12 MESES"},
]


# =====================================================================
# RUIDO OCR CONTROLADO (SUBSTITUCIONES TIPICAS DE OCR)
# =====================================================================

def introduce_ocr_noise(text: str, noise_level: float = 0.05) -> str:
    """Simulates common OCR character confusion while preserving general shape."""
    if noise_level <= 0.0:
        return text
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < noise_level:
            c = chars[i]
            if c == 'O': chars[i] = '0'
            elif c == '0' and random.random() < 0.3: chars[i] = 'O'
            elif c in ('l', 'I', '|'): chars[i] = '1'
            elif c == '1' and random.random() < 0.3: chars[i] = 'l'
            elif c == 'S': chars[i] = '5'
            elif c == '5' and random.random() < 0.2: chars[i] = 'S'
            elif c == 'B' and random.random() < 0.2: chars[i] = '8'
    return "".join(chars)


# =====================================================================
# GENERADORES DE FACTURAS POR TIPO DE NEGOCIO
# =====================================================================

class InvoiceBuilder:
    """Helper to assemble lines and track labeled tokens cleanly."""
    def __init__(self):
        self.lines: List[List[Tuple[str, str]]] = []

    def add_line(self, tokens: List[Tuple[str, str]] | List[str]):
        line_tokens = []
        for item in tokens:
            if isinstance(item, tuple):
                val, lbl = item
                subwords = str(val).split()
                for w in subwords:
                    line_tokens.append((w, lbl))
            else:
                subwords = str(item).split()
                for w in subwords:
                    line_tokens.append((w, "irrelevant"))
        if line_tokens:
            self.lines.append(line_tokens)

    def render(self, noise_level: float = 0.0) -> Tuple[str, List[dict]]:
        text_lines = []
        flat_labels = []
        for line_tokens in self.lines:
            words = []
            for word, label in line_tokens:
                mutated = introduce_ocr_noise(word, noise_level) if label == "irrelevant" else word
                words.append(mutated)
                flat_labels.append({"token": mutated, "label": label})
            text_lines.append(" ".join(words))
        return "\n".join(text_lines), flat_labels


def random_date_str() -> Tuple[str, str, str]:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2024, 2026)
    fmt = random.choice(["DD/MM/YYYY", "YYYY-MM-DD", "YYYYMMDD"])
    if fmt == "DD/MM/YYYY":
        d_str = f"{day:02d}/{month:02d}/{year}"
    elif fmt == "YYYY-MM-DD":
        d_str = f"{year}-{month:02d}-{day:02d}"
    else:
        d_str = f"{year}{month:02d}{day:02d}"
    vence_str = f"31/12/{year}"
    return d_str, vence_str, f"{year}{month:02d}{day:02d}"


def random_ncf(prefer_ecf: bool = False, credit_fiscal: bool = True) -> str:
    seq = random.randint(10000000, 99999999)
    if prefer_ecf:
        code = "E31" if credit_fiscal else "E32"
        return f"{code}{seq:010d}"
    else:
        code = "B01" if credit_fiscal else "B02"
        return f"{code}{seq:08d}"


def generate_combustible_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    """Generates real-world fuel station invoice (Shell, Total, Sunix, Propagas, Tropigas)."""
    b = InvoiceBuilder()
    station = random.choice(COMBUSTIBLE_STATIONS)
    is_glp = station.get("fuel_type") == "glp"
    prefer_ecf = random.choice([True, False])
    credit_fiscal = random.choice([True, True, False])
    
    date_str, vence_str, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=prefer_ecf, credit_fiscal=credit_fiscal)
    rnc_raw = station["rnc"]
    rnc_clean = re.sub(r"\D", "", rnc_raw)
    
    # Header
    b.add_line([station["name"]])
    b.add_line([station["company"]])
    rnc_lbl = random.choice(["RNC:", "R.N.C.:", "RNC"])
    b.add_line([(rnc_lbl, "irrelevant"), (rnc_raw, "rnc_cedula")])
    b.add_line([station["address"]])
    b.add_line([f"Tel: {station['phone']}"])
    b.add_line(["------------------------------------------------"])
    
    # NCF & Dates
    ncf_lbl = "FACTURA DE CREDITO FISCAL (e-CF):" if prefer_ecf else "FACTURA DE CREDITO FISCAL:"
    if not credit_fiscal:
        ncf_lbl = "FACTURA DE CONSUMO:"
    b.add_line([(ncf_lbl, "irrelevant"), (ncf_str, "ncf")])
    if not prefer_ecf:
        b.add_line(["VENCE:", vence_str])
    fecha_lbl = random.choice(["FECHA:", "FECHA EMISION:", "F. COMP:"])
    b.add_line([(fecha_lbl, "irrelevant"), (date_str, "fecha_comprobante"), "HORA: 09:24:12 AM"])
    
    # Fuel dispatch details
    bomba = f"{random.randint(1, 16):02d}"
    manguera = f"{random.randint(1, 4):02d}"
    trans = f"{random.randint(1000, 99999):05d}"
    b.add_line(["TRANSACCION:", trans, "DISPENSADOR/BOMBA:", bomba, "MANGUERA:", manguera])
    b.add_line(["ATENDIDO POR: BOMBERO", f"{random.randint(10, 99)}"])
    b.add_line(["------------------------------------------------"])
    
    # Client info if B01/E31
    if credit_fiscal:
        b.add_line(["DATOS DEL CLIENTE / ADQUIRENTE:"])
        client_rnc = f"13{random.randint(1000000, 9999999)}"
        b.add_line(["RNC CLIENTE:", client_rnc])
        b.add_line(["RAZON SOCIAL: TRANSPORTES DOMINICANOS, S.R.L."])
        b.add_line(["------------------------------------------------"])
        
    # Product line
    product = random.choice(GLP_PRODUCTS if is_glp else FUEL_PRODUCTS)
    gallons = round(random.uniform(5.0, 45.0), 3)
    total_val = round(gallons * product["price"], 2)
    
    amt_fmt = "{:,.2f}"
    price_str = amt_fmt.format(product["price"])
    total_str = amt_fmt.format(total_val)
    gal_str = f"{gallons:.3f} {product['unit']}"
    
    b.add_line(["CANTIDAD", "DESCRIPCION", "PRECIO", "TOTAL"])
    b.add_line([gal_str, product["desc"], price_str, total_str])
    b.add_line(["------------------------------------------------"])
    
    # Subtotal and EXEMPT ITBIS (Ley 112-00)
    b.add_line(["SUBTOTAL EXENTO (LEY 112-00):", f"RD$ {total_str}"])
    itbis_zero = random.choice(["0.00", "0.00", "EXENTO"])
    if itbis_zero == "0.00":
        b.add_line([("ITBIS (0%):", "irrelevant"), ("RD$", "irrelevant"), (itbis_zero, "itbis_facturado")])
    else:
        b.add_line(["ITBIS:", "EXENTO"])
        
    b.add_line(["------------------------------------------------"])
    total_lbl = random.choice(["TOTAL A PAGAR:", "TOTAL FACTURADO:", "TOTAL NETO:"])
    b.add_line([(total_lbl, "irrelevant"), ("RD$", "irrelevant"), (total_str, "monto_total")])
    b.add_line(["------------------------------------------------"])
    
    # Payment voucher
    if random.choice([True, False]):
        b.add_line(["FORMA DE PAGO: TARJETA DE CREDITO AZUL"])
        b.add_line(["TERMINAL: AZ09148", "LOTE: 0041", "APROBACION:", f"{random.randint(100000, 999999)}"])
        b.add_line(["TARJETA: VISA", f"************{random.randint(1000, 9999)}"])
    else:
        b.add_line(["FORMA DE PAGO: EFECTIVO"])
        
    if prefer_ecf:
        b.add_line(["[ CODIGO QR DGII ] Codigo Seguridad:", f"{random.randint(100000, 999999)}"])
        b.add_line(["Documento fiscal electronico validado por la DGII"])
    b.add_line(["¡Gracias por su compra! Pase buenas."])
    
    return b.render(noise_level)


def generate_hardware_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    """Generates real-world hardware store invoice (Ochoa, Americana, Bellón, etc.)."""
    b = InvoiceBuilder()
    store = random.choice(FERRETERIA_STORES)
    prefer_ecf = random.choice([True, False])
    
    date_str, vence_str, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=prefer_ecf, credit_fiscal=True)
    rnc_raw = store["rnc"]
    
    b.add_line([store["name"]])
    b.add_line([store["company"]])
    b.add_line([("RNC:", "irrelevant"), (rnc_raw, "rnc_cedula")])
    b.add_line([store["address"]])
    b.add_line([f"Tel: {store['phone']}"])
    b.add_line(["================================================"])
    
    b.add_line([("FACTURA DE CREDITO FISCAL:", "irrelevant"), (ncf_str, "ncf")])
    if not prefer_ecf:
        b.add_line(["VENCE:", vence_str])
    b.add_line([("FECHA:", "irrelevant"), (date_str, "fecha_comprobante"), "HORA: 11:15:40 AM"])
    b.add_line(["CONDUCE NO.:", f"CD-{random.randint(10000, 99999)}", "CONDICION: CREDITO 30 DIAS"])
    b.add_line(["CLIENTE: CONSTRUCTORA DEL CIBAO, S.A.", "RNC CLIENTE: 1-30-98412-1"])
    b.add_line(["------------------------------------------------"])
    
    # Items
    items = random.sample(HARDWARE_PRODUCTS, random.randint(2, 4))
    subtotal = 0.0
    b.add_line(["CANT", "UNIDAD", "DESCRIPCION", "PRECIO", "TOTAL"])
    for item in items:
        qty = random.randint(1, 20)
        line_tot = round(qty * item["price"], 2)
        subtotal += line_tot
        b.add_line([f"{qty}.00", item["unit"], item["desc"], f"{item['price']:,.2f}", f"{line_tot:,.2f}"])
        
    itbis_val = round(subtotal * 0.18, 2)
    total_val = round(subtotal + itbis_val, 2)
    
    b.add_line(["------------------------------------------------"])
    b.add_line(["SUB-TOTAL:", f"RD$ {subtotal:,.2f}"])
    b.add_line([("ITBIS (18%):", "irrelevant"), ("RD$", "irrelevant"), (f"{itbis_val:,.2f}", "itbis_facturado")])
    b.add_line(["------------------------------------------------"])
    b.add_line([("TOTAL A PAGAR:", "irrelevant"), ("RD$", "irrelevant"), (f"{total_val:,.2f}", "monto_total")])
    b.add_line(["------------------------------------------------"])
    b.add_line(["FORMA DE PAGO: CHEQUE / TRANSFERENCIA"])
    b.add_line(["RECIBIDO CONFORME (FIRMA/SELLO): ___________________"])
    return b.render(noise_level)


def generate_tech_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    """Generates computer and electronics invoice (Cecomsa, Omega Tech, Corripio, etc.)."""
    b = InvoiceBuilder()
    store = random.choice(TECH_STORES)
    prefer_ecf = random.choice([True, False])
    
    date_str, vence_str, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=prefer_ecf, credit_fiscal=True)
    rnc_raw = store["rnc"]
    
    b.add_line([store["name"]])
    b.add_line([store["company"]])
    b.add_line([("RNC:", "irrelevant"), (rnc_raw, "rnc_cedula")])
    b.add_line([store["address"]])
    b.add_line([f"Tel: {store['phone']}"])
    b.add_line(["================================================"])
    
    b.add_line([("FACTURA DE CREDITO FISCAL (e-CF):" if prefer_ecf else "FACTURA CREDITO FISCAL:", "irrelevant"), (ncf_str, "ncf")])
    b.add_line([("FECHA:", "irrelevant"), (date_str, "fecha_comprobante"), "CAJA: 02", "VENDEDOR: 041"])
    b.add_line(["ADQUIRENTE:", "SOLUCIONES TECNOLOGICAS DEL CARIBE SRL", "RNC: 1-31-04928-1"])
    b.add_line(["------------------------------------------------"])
    
    items = random.sample(TECH_PRODUCTS, random.randint(1, 3))
    subtotal = 0.0
    b.add_line(["CANT", "DESCRIPCION", "PRECIO", "IMPORTE"])
    for item in items:
        qty = random.randint(1, 3)
        line_tot = round(qty * item["price"], 2)
        subtotal += line_tot
        b.add_line([str(qty), item["desc"], f"{item['price']:,.2f}", f"{line_tot:,.2f}"])
        if item.get("has_serial"):
            serial = f"S/N: {random.choice(['SN', 'DP', 'CN', '309'])}{random.randint(100000, 999999)}"
            b.add_line(["  ", serial, f"GARANTIA: {item['warranty']}"])
            
    itbis_val = round(subtotal * 0.18, 2)
    total_val = round(subtotal + itbis_val, 2)
    
    b.add_line(["------------------------------------------------"])
    b.add_line(["SUBTOTAL NETO:", f"RD$ {subtotal:,.2f}"])
    b.add_line([("18% ITBIS:", "irrelevant"), ("RD$", "irrelevant"), (f"{itbis_val:,.2f}", "itbis_facturado")])
    b.add_line(["------------------------------------------------"])
    b.add_line([("TOTAL A PAGAR:", "irrelevant"), ("RD$", "irrelevant"), (f"{total_val:,.2f}", "monto_total")])
    b.add_line(["------------------------------------------------"])
    b.add_line(["FORMA DE PAGO: TRANSFERENCIA BANCO POPULAR DOMINICANO"])
    b.add_line(["GARANTIA: REQUISITO PRESENTAR FACTURA ORIGINAL Y EMPAQUE LEY 358-05"])
    return b.render(noise_level)


def generate_supermarket_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    b = InvoiceBuilder()
    store = random.choice(SUPERMARKETS)
    prefer_ecf = random.choice([True, True, False])
    
    date_str, _, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=prefer_ecf, credit_fiscal=False)
    
    b.add_line([store["name"]])
    b.add_line([store["company"]])
    b.add_line([("RNC:", "irrelevant"), (store["rnc"], "rnc_cedula")])
    b.add_line([store["address"]])
    b.add_line(["================================================"])
    b.add_line([("FACTURA CONSUMIDOR FINAL (e-CF):", "irrelevant"), (ncf_str, "ncf")])
    b.add_line([("FECHA:", "irrelevant"), (date_str, "fecha_comprobante"), "CAJERO: 014", "POS: 03"])
    b.add_line(["------------------------------------------------"])
    
    sub_exento = round(random.uniform(300, 1500), 2)
    sub_gravado = round(random.uniform(500, 2500), 2)
    itbis_val = round(sub_gravado * 0.18, 2)
    total_val = round(sub_exento + sub_gravado + itbis_val, 2)
    
    b.add_line(["1.00 ARROZ CAMPOS SUP 10 LB", "E", f"{sub_exento:,.2f}"])
    b.add_line(["2.00 JABON EN POLVO ARIEL 1.8KG", "18%", f"{sub_gravado:,.2f}"])
    b.add_line(["------------------------------------------------"])
    b.add_line(["SUBTOTAL EXENTO:", f"RD$ {sub_exento:,.2f}"])
    b.add_line(["SUBTOTAL GRAVADO 18%:", f"RD$ {sub_gravado:,.2f}"])
    b.add_line([("TOTAL ITBIS:", "irrelevant"), ("RD$", "irrelevant"), (f"{itbis_val:,.2f}", "itbis_facturado")])
    b.add_line(["------------------------------------------------"])
    b.add_line([("TOTAL A PAGAR:", "irrelevant"), ("RD$", "irrelevant"), (f"{total_val:,.2f}", "monto_total")])
    b.add_line(["CARDNET - COMPRA VISA ************4910 APROB: 849120"])
    return b.render(noise_level)


def generate_restaurant_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    b = InvoiceBuilder()
    store = random.choice(RESTAURANTS)
    date_str, _, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=True, credit_fiscal=False)
    
    b.add_line([store["name"]])
    b.add_line([store["company"]])
    b.add_line([("RNC:", "irrelevant"), (store["rnc"], "rnc_cedula")])
    b.add_line([store["address"]])
    b.add_line(["================================================"])
    b.add_line([("FACTURA CONSUMO e-CF:", "irrelevant"), (ncf_str, "ncf")])
    b.add_line([("FECHA:", "irrelevant"), (date_str, "fecha_comprobante"), "MESA: 12", "MOZO: 04"])
    b.add_line(["------------------------------------------------"])
    
    subtotal = round(random.uniform(800, 6000), 2)
    itbis_val = round(subtotal * 0.18, 2)
    propina_ley = round(subtotal * 0.10, 2)
    total_val = round(subtotal + itbis_val + propina_ley, 2)
    
    b.add_line(["1 CONSUMO ALIMENTOS Y BEBIDAS", f"{subtotal:,.2f}"])
    b.add_line(["------------------------------------------------"])
    b.add_line(["SUBTOTAL CONSUMO:", f"RD$ {subtotal:,.2f}"])
    b.add_line([("18% ITBIS:", "irrelevant"), ("RD$", "irrelevant"), (f"{itbis_val:,.2f}", "itbis_facturado")])
    b.add_line(["10% PROPINA LEGAL (LEY 5432):", f"RD$ {propina_ley:,.2f}"])
    b.add_line(["------------------------------------------------"])
    b.add_line([("TOTAL FACTURADO:", "irrelevant"), ("RD$", "irrelevant"), (f"{total_val:,.2f}", "monto_total")])
    b.add_line(["FORMA DE PAGO: TARJETA AZUL - LOTE: 0014 APROB: 948123"])
    return b.render(noise_level)


def generate_pharmacy_invoice(noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    b = InvoiceBuilder()
    store = random.choice(PHARMACIES)
    date_str, _, _ = random_date_str()
    ncf_str = random_ncf(prefer_ecf=False, credit_fiscal=False)
    
    b.add_line([store["name"]])
    b.add_line([store["company"]])
    b.add_line([("RNC:", "irrelevant"), (store["rnc"], "rnc_cedula")])
    b.add_line([store["address"]])
    b.add_line(["================================================"])
    b.add_line([("FACTURA CONSUMIDOR FINAL:", "irrelevant"), (ncf_str, "ncf")])
    b.add_line([("FECHA:", "irrelevant"), (date_str, "fecha_comprobante"), "CAJA: 01"])
    b.add_line(["------------------------------------------------"])
    
    meds = round(random.uniform(400, 2000), 2)
    misc = round(random.uniform(200, 1000), 2)
    itbis_val = round(misc * 0.18, 2)
    total_val = round(meds + misc + itbis_val, 2)
    
    b.add_line(["1 MEDICAMENTO RECETADO", "E", f"{meds:,.2f}"])
    b.add_line(["1 PROTECTOR SOLAR / CUIDADO PERSONAL", "18%", f"{misc:,.2f}"])
    b.add_line(["------------------------------------------------"])
    b.add_line(["TOTAL MEDICAMENTOS EXENTOS:", f"RD$ {meds:,.2f}"])
    b.add_line(["MISCELANEOS GRAVADOS:", f"RD$ {misc:,.2f}"])
    b.add_line([("ITBIS 18%:", "irrelevant"), ("RD$", "irrelevant"), (f"{itbis_val:,.2f}", "itbis_facturado")])
    b.add_line(["------------------------------------------------"])
    b.add_line([("TOTAL A PAGAR:", "irrelevant"), ("RD$", "irrelevant"), (f"{total_val:,.2f}", "monto_total")])
    b.add_line(["AFILIADO ARS HUMANO PLAN BASICO AUT: 849120"])
    return b.render(noise_level)


# =====================================================================
# DATASET GENERATOR CON ÉNFASIS EN COMBUSTIBLE, FERRETERÍA Y TECNOLOGÍA
# =====================================================================

def generate_invoice(sector: str | None = None, noise_level: float = 0.05) -> Tuple[str, List[dict]]:
    """Generates an invoice from one of the sectors, heavily weighting user priorities."""
    if not sector:
        # Weighting: 45% Combustible (GLP/Gasolina), 20% Ferretería, 20% Cómputo/Electrónica, 15% others
        sector = random.choices(
            ["combustible", "ferreteria", "tech", "supermercado", "restaurante", "farmacia"],
            weights=[45, 20, 20, 5, 5, 5],
            k=1
        )[0]
        
    if sector == "combustible":
        return generate_combustible_invoice(noise_level)
    elif sector == "ferreteria":
        return generate_hardware_invoice(noise_level)
    elif sector == "tech":
        return generate_tech_invoice(noise_level)
    elif sector == "supermercado":
        return generate_supermarket_invoice(noise_level)
    elif sector == "restaurante":
        return generate_restaurant_invoice(noise_level)
    else:
        return generate_pharmacy_invoice(noise_level)


def generate_dataset(n: int = 3500, noise_levels: List[float] = [0.0, 0.03, 0.06, 0.10]) -> Tuple[List[str], List[List[dict]]]:
    """Generates n diverse invoices covering all Dominican real company profiles."""
    random.seed(42)
    texts = []
    labels_list = []
    for _ in range(n):
        noise = random.choice(noise_levels)
        text, labels = generate_invoice(noise_level=noise)
        texts.append(text)
        labels_list.append(labels)
    return texts, labels_list


def save_dataset(texts: List[str], labels_list: List[List[dict]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels_list):
            f.write(json.dumps({"text": t, "labels": l}, ensure_ascii=False) + "\n")
