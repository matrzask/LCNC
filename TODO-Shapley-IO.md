**Atrybucja Shapleya w IO**

Poprzednio badaliśmy prompt-to-feature, gdzie były rozważane z jednej stronyfragmenty promptu a z drugiej wpływ na funkcjonalności rozumiane jako funkcje.

Chcemy się rozszerzyć na inne domeny, co jest możliwe.

1. **Inżynieria wymagania**

Prompt:  
P1: „Pacjent może umówić wizytę".  
P2: „Wizytę można odwołać najpóźniej 24 godziny przed terminem".  
P3: „Po rezerwacji pacjent dostaje e-mail".

Wynik generowania LLM:  
R1: wymaganie rezerwacji wizyty; R2: reguła odwołania z limitem 24 godzin; R3: wymaganie wysłania e-maila.

2. **Architektura**

Prompt:  
P1: „System ma obsługiwać 500 równoczesnych użytkowników".  
P2: „Dane pacjentów muszą być szyfrowane".  
P3: „Potwierdzenie wizyty jest wysyłane e-mailem".

Wynik generowania LLM: diagram komponentów, z nazwanymi komponentami i połączeniami.

3. **Testowanie**

Prompt:  
P1: „Rezerwacja zapisuje wizytę".  
P2: „Odwołanie jest dozwolone do 24 godzin przed wizytą".  
P3: „Po rezerwacji pacjent dostaje e-mail".

Wynik generowania: przypadki testowe, np. w postaci testów jednostkowych.

- T1 — booking_creates_appointment: po rezerwacji wizyta jest zapisana.
- T2 — cancellation_at_24_hours_is_accepted: odwołanie dokładnie 24 godziny przed wizytą zostaje przyjęte.
- T3 — cancellation_at_23_hours_is_rejected: odwołanie 23 godziny przed wizytą zostaje odrzucone.
- T4 — booking_sends_confirmation_email: po rezerwacji zostaje wysłany e-mail.

4. **System agentowy**

Może dla wieloagentowego systemu do modyfikacji kodu taki eksperyment.

System agentowy

- Planner rozbija żądanie na zadania.
- Coder czyta i zmienia kod.
- Tester dodaje testy i uruchamia je.

Repozytorium zawiera endpoint zgłoszeń, moduł eksportu CSV i testy.

Prompt

- P1: Dodaj do endpointu zgłoszeń eksport do CSV.
- P2: Uwzględnij w CSV status i osobę przypisaną.
- P3: Pomiń zgłoszenia poufne.

Co system agentowy ma wygenerować

- Planner: zadanie „dodać eksport CSV", zadanie „dodać kolumny status i osoba przypisana", zadanie „odfiltrować zgłoszenia poufne".
- Coder: wywołanie odczytu modułu eksportu, następnie jego edycję. Wynikiem edycji jest nagłówek id,title,status,assignee i filtr wykluczający zgłoszenia poufne.
- Tester: test obecności kolumn oraz test potwierdzający, że zgłoszenie poufne nie trafia do CSV; następnie wywołanie testów.

Co tutaj mierzy Shapley

Fragmenty P1–P3 są przypisywane do konkretnych działań i wyników agentów, np.:

- P1 → zadanie Plannera dotyczące eksportu; zmiana Coder'a dodająca obsługę CSV.
- P2 → zadanie Plannera dotyczące kolumn; zmiana nagłówka CSV.
- P3 → zadanie Plannera dotyczące poufności; filtr w kodzie; test poufności.

System uruchamia się od tego samego stanu repozytorium dla wszystkich 8 podzbiorów promptu. Dla każdego celu, np. „Coder dodał filtr poufności", sprawdza się, czy ten wynik powstał poprawnie. Shapley oblicza wkład każdego fragmentu promptu w każde konkretne działanie lub wytworzony element systemu agentowego.