# TODO — Comment utiliser le script

## 1. Choisir ton timeframe

- [ ] **1W** = Position Weekly  
  Objectif : rester en position plusieurs semaines/mois.

- [ ] **1D** = Swing Daily  
  Objectif : trade de quelques jours à quelques semaines.

- [ ] **4H** = Swing 4H  
  Objectif : trade plus court, généralement quelques heures à quelques jours.

Le script change automatiquement de mode selon le timeframe TradingView.

---

# 2. Si je n'ai AUCUNE position

La ligne principale à regarder est :

**`Nouvelle entrée`**

Ne te base pas sur `Gestion position` si tu n'as pas pris le précédent BUY.

## 🟢 BUY NOW

Si tu vois :

- `BUY NOW · REVERSAL`
- `BUY NOW · BREAKOUT`
- `BUY NOW · CONT`

alors une entrée est autorisée par le modèle.

Avant d'acheter, vérifier aussi :

- [ ] `Sécurité live = OK · LIVE`
- [ ] l'écart avec `Prix signal entrée` n'est pas trop important
- [ ] le prix n'est pas en train de chuter brutalement au moment où tu passes l'ordre

### Signification des BUY

**BUY R = REVERSAL**  
Le marché était faible et commence à se retourner.

**BUY B = BREAKOUT**  
Le marché casse une résistance / un plus haut avec momentum.

**BUY C = CONTINUATION**  
La tendance est déjà haussière, le marché fait une respiration/pullback puis repart.

---

# 3. Quand NE PAS acheter

## 🟠 WATCH

`WATCH · SETUP`

→ Quelque chose se prépare mais **pas encore de BUY**.

- [ ] attendre la prochaine clôture
- [ ] ne pas anticiper le signal

---

## 🟠 WAIT · PULLBACK

La tendance peut être bonne mais le prix est déjà trop haut.

→ **Ne pas courir derrière le marché.**

Attendre :

- retour EMA20
- consolidation
- nouveau BUY C
- ou nouveau breakout propre

---

## 🔴 AVOID · FALLING

Le marché est actuellement en train de se dégrader.

→ **Pas d'achat.**

Même si BTC semble "moins cher", le script considère que la baisse peut continuer.

---

## 🔴 WAIT · LIVE FALLING

Le dernier chandelier clôturé donnait éventuellement un setup intéressant, mais **la bougie actuellement ouverte est en train de tomber**.

→ attendre.

Le filtre live peut bloquer un BUY mais il ne peut jamais créer un BUY.

---

## 🔴 WAIT · MISSED / CHASE

Un BUY a été donné précédemment mais tu arrives trop tard.

Exemple :

> BUY à 65 000 $  
> BTC est déjà à 69 000 $

→ **ne pas acheter simplement parce que le modèle est toujours IN.**

Attendre le prochain setup.

---

# 4. Très important : HOLD ≠ BUY

Si le dashboard affiche :

**`Gestion position = HOLD`**

cela veut dire :

> Le modèle a acheté AVANT et recommande de conserver cette ancienne position.

Cela ne veut PAS dire :

> Acheter maintenant.

### Exemple

Tu vois :

> Position modèle : `IN · CONTINUATION`  
> Gestion position : `HOLD`  
> Nouvelle entrée : `WAIT · PULLBACK`

Si tu avais acheté lors du signal :

→ **HOLD**

Si tu n'as aucune position :

→ **WAIT**, tu n'achètes pas.

---

# 5. Si j'ai déjà acheté

À partir de là, tu regardes principalement :

**`Gestion position`**

---

## 🔵 HOLD

→ conserver.

Le moteur considère que la tendance reste suffisamment saine.

- [ ] ne pas vendre juste parce que BTC a déjà beaucoup monté
- [ ] ne pas refaire un BUY
- [ ] attendre TP / SELL

---

## 🟠 HOLD · TP WATCH

Tu es toujours en tendance haussière mais :

- gain déjà important
- RSI élevé
- MACD historiquement élevé
- prix fortement étendu

→ **on ne SELL pas encore automatiquement**, mais on se prépare à sécuriser.

Pour le Paper Trading :

- [ ] noter `TP WATCH`
- [ ] surveiller particulièrement les prochaines clôtures

---

# 6. TP1

Quand le script affiche :

**`TP1`**

→ première vraie zone de prise de bénéfices.

Pour notre test, on peut utiliser :

**25 % de la position**

Exemple :

Tu as :

> 0,10 BTC

TP1 :

> vendre 0,025 BTC

et conserver :

> 0,075 BTC

Le but est de :

> sécuriser une partie du bénéfice  
> tout en laissant courir la tendance.

---

# 7. Après TP1

Le reste de la position continue.

Tu peux retrouver :

**HOLD**

→ conserver le reste.

Il n'y aura normalement qu'un **TP1 par trade**, pas 5 TP successifs.

---

# 8. SELL

Si le script affiche :

**`SELL`**

→ fermeture de la position restante.

Le SELL peut venir notamment de :

- retournement MACD
- perte du momentum
- perte EMA20
- détérioration importante de tendance
- perte EMA50
- combinaison de plusieurs éléments

### Donc

`TP WATCH` ≠ SELL  
`TP1` ≠ forcément tout vendre  
`SELL` = vraie sortie du modèle

---

# 9. Workflow très simple

## Si je suis OUT

Regarder :

**Nouvelle entrée**

### Si :

🟢 `BUY NOW`
+
🟢 `Sécurité live = OK`

→ **BUY**

### Si :

🟠 `WATCH`

→ attendre

### Si :

🟠 `WAIT PULLBACK`

→ attendre

### Si :

🔴 `FALLING`

→ ne pas acheter

### Si :

🔴 `MISSED / CHASE`

→ signal raté, attendre le prochain

---

# 10. Si je suis IN

Regarder :

**Gestion position**

### `HOLD`
→ conserver

### `HOLD · TP WATCH`
→ conserver mais surveiller fortement

### `TP1`
→ prendre une partie des bénéfices

### `SELL`
→ fermer le reste

---

# 11. Pour ton Paper Trading

À chaque clôture de bougie :

### 4H

Vérifier toutes les **4 heures** si tu testes activement ce mode.

### Daily

Vérifier après chaque **clôture Daily**.

### Weekly

Un vrai nouveau call Weekly n'est validé qu'à la **clôture de la semaine**.

---

# 12. Règle essentielle pendant les 7 jours

- [ ] Ne pas modifier le Pine Script
- [ ] Ne pas anticiper les BUY
- [ ] Ne pas transformer HOLD en BUY
- [ ] Ne pas acheter un signal déjà trop éloigné
- [ ] Respecter TP1 / SELL comme indiqué
- [ ] Noter chaque anomalie dans l'Excel
- [ ] Noter également les moments où **toi tu aurais fait autrement**
- [ ] Garder des screenshots des cas bizarres

---

# MÉMO EXPRESS

**Je n'ai pas de position :**

`BUY NOW` → acheter  
`WATCH` → surveiller  
`WAIT` → attendre  
`AVOID` → ne pas acheter  
`HOLD` → ignorer, car ce HOLD concerne une ancienne entrée

**J'ai une position :**

`HOLD` → garder  
`TP WATCH` → préparer la sécurisation  
`TP1` → prendre une partie des bénéfices  
`SELL` → sortir

### Règle numéro 1

**Pas de `BUY NOW` = pas de nouvel achat.**