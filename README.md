import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Clases y funciones comunes ---

class FunctionPlotter:
    def __init__(self):
        self.history = []
        self.current_index = -1

    def add_plot(self, f_expr, a, b):
        self.history.append((f_expr, a, b))
        self.current_index = len(self.history) - 1

    def get_plot(self, index):
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

def plot_function(f_expr, a, b, y_lim):
    plt.close('all')
    x = sp.Symbol('x')
    f_lambdified = sp.lambdify(x, f_expr, 'numpy')
    num_points = 1000
    x_vals = np.linspace(a, b, num_points)
    y_vals = np.array([f_lambdified(xi) for xi in x_vals])
    
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=str(f_expr), color='violet')
    ax.axhline(0, color='white', linewidth=0.8)
    ax.axvline(0, color='white', linewidth=0.8)
    ax.legend()
    ax.set_facecolor("#2E2E2E")
    fig.patch.set_facecolor("#1E1E1E")
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.set_ylim(-y_lim, y_lim)
    return fig

def clean_function_input(expr):
    expr = expr.replace(" ", "")
    # Si se detecta 'x' sin multiplicación, se puede ajustar
    if "x" in expr and "*x" not in expr:
        expr = expr.replace("x", "*x")
    return expr

# --- Métodos numéricos ---

def bisection_method(f, a, b, tol, max_iter, table, root_label):
    if f(a) * f(b) >= 0:
        messagebox.showerror("Error", "El intervalo no cambia de signo")
        return None
    
    root_found = None
    table.delete(*table.get_children())
    for i in range(max_iter):
        c = (a + b) / 2
        table.insert("", "end", values=(i + 1, a, b, c, f(c)))
        
        if abs(f(c)) <= tol or (b - a) / 2 <= tol:
            root_found = c
            break
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    if root_found is not None:
        root_label.config(text=f"{root_found:.6f}", bg="#2E2E2E")
    else:
        root_label.config(text="No encontrada", bg="#2E2E2E")
    return root_found

def secant_method(f, x0, x1, tol, max_iter, table, root_label):
    results = []
    table.delete(*table.get_children())
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            messagebox.showerror("Error", "División por cero en la iteración " + str(i+1))
            return None
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        results.append((i, x_new, fx1))
        table.insert("", "end", values=(i + 1, x0, x1, x_new, fx1))
        if abs(x_new - x1) < tol:
            root_label.config(text=f"{x_new:.6f}", bg="#2E2E2E")
            return x_new
        x0, x1 = x1, x_new
    messagebox.showerror("Error", "No se encontró raíz en el número máximo de iteraciones")
    return None

def newton_raphson_method(f, df, x0, tol, max_iter, table, root_label):

    table.delete(*table.get_children())
    x_n = x0
    for i in range(max_iter):
        fx_n = f(x_n)
        dfx_n = df(x_n)
        if dfx_n == 0:
            messagebox.showerror("Error", f"Derivada igual a cero en la iteración {i+1}")
            return None
        x_nuevo = x_n - fx_n / dfx_n
        
        table.insert("", "end", values=(
            i + 1, 
            f"{x_n:.12f}", 
            f"{x_nuevo:.12f}", 
            f"{fx_n:.12e}"
        ))
        
        if abs(x_nuevo - x_n) < tol:
            root_label.config(text=f"{x_nuevo:.12}", bg="#2E2E2E")
            return x_nuevo
        
        x_n = x_nuevo
        
    messagebox.showerror("Error", "No se encontró raíz en el número máximo de iteraciones")
    return None

# --- Funciones para cada pestaña ---

def calculate_bisection():
    try:
        # Limpiar gráfico
        for widget in graph_frame_bi.winfo_children():
            widget.destroy()
        
        expr = function_entry_bi.get()
        expr = clean_function_input(expr)
        f_expr = sp.sympify(expr)
        x = sp.Symbol('x')
        f = sp.lambdify(x, f_expr, 'numpy')
        
        a = float(entry_a_bi.get())
        b = float(entry_b_bi.get())
        tol = float(entry_tol_bi.get())
        max_iter = int(entry_iter_bi.get())
        y_lim = float(entry_c_bi.get())
        
        root_value_bi.config(text="Calculando...")
        root = bisection_method(f, a, b, tol, max_iter, table_bi, root_value_bi)
        
        plotter.add_plot(f_expr, a, b)
        fig = plot_function(f_expr, a, b, y_lim)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame_bi)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def calculate_secant():
    try:
        for widget in graph_frame_sec.winfo_children():
            widget.destroy()
        
        expr = function_entry_sec.get()
        expr = clean_function_input(expr)
        f_expr = sp.sympify(expr)
        x = sp.Symbol('x')
        f = sp.lambdify(x, f_expr, 'numpy')
        
        x0 = float(entry_x0_sec.get())
        x1 = float(entry_x1_sec.get())
        tol = float(entry_tol_sec.get())
        max_iter = int(entry_iter_sec.get())
        y_lim = float(entry_c_sec.get())
        
        root_value_sec.config(text="Calculando...")
        root = secant_method(f, x0, x1, tol, max_iter, table_sec, root_value_sec)
        
        # Para graficar, se usa un intervalo entre min(x0,x1) y max(x0,x1) con un margen extra
        left = min(x0, x1) - 1
        right = max(x0, x1) + 1
        plotter.add_plot(f_expr, left, right)
        fig = plot_function(f_expr, left, right, y_lim)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame_sec)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def calculate_newton():
    try:
        for widget in graph_frame_newt.winfo_children():
            widget.destroy()
        
        expr = function_entry_newt.get()
        expr = clean_function_input(expr)
        f_expr = sp.sympify(expr)
        x = sp.Symbol('x')
        f = sp.lambdify(x, f_expr, 'numpy')
        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, 'numpy')
        
        x0 = float(entry_x0_newt.get())
        tol = float(entry_tol_newt.get())
        max_iter = int(entry_iter_newt.get())
        y_lim = float(entry_c_newt.get())
        
        root_value_newt.config(text="Calculando...")
        root = newton_raphson_method(f, df, x0, tol, max_iter, table_newt, root_value_newt)
        
        # Para el gráfico, se define un rango centrado en x0
        left = x0 - 5
        right = x0 + 5
        plotter.add_plot(f_expr, left, right)
        fig = plot_function(f_expr, left, right, y_lim)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame_newt)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# --- Configuración de la ventana principal y Notebook ---

plotter = FunctionPlotter()

root = tk.Tk()
root.title("Calculadora de Métodos Numéricos")
root.configure(bg="#1E1E1E")
root.geometry("1200x800")

main_frame = tk.Frame(root, bg="#1E1E1E")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

notebook = ttk.Notebook(main_frame)
notebook.pack(fill=tk.BOTH, expand=True)

# ----- Pestaña Bisección -----
frame_bi = tk.Frame(notebook, bg="#1E1E1E")
notebook.add(frame_bi, text="Bisección")

# Frame de entradas
input_frame_bi = tk.Frame(frame_bi, bg="#1E1E1E")
input_frame_bi.pack(fill=tk.X, pady=5)

tk.Label(input_frame_bi, text="Altura del gráfico en Y:", fg="white", bg="#1E1E1E").grid(row=0, column=0, sticky='w')
entry_c_bi = tk.Entry(input_frame_bi, width=10, bg="#2E2E2E", fg="white")
entry_c_bi.grid(row=0, column=1, padx=5)

tk.Label(input_frame_bi, text="Función f(x):", fg="white", bg="#1E1E1E").grid(row=1, column=0, sticky='w')
function_entry_bi = tk.Entry(input_frame_bi, width=30, bg="#2E2E2E", fg="white")
function_entry_bi.grid(row=1, column=1, padx=5)

tk.Label(input_frame_bi, text="a:", fg="white", bg="#1E1E1E").grid(row=2, column=0, sticky='w')
entry_a_bi = tk.Entry(input_frame_bi, width=10, bg="#2E2E2E", fg="white")
entry_a_bi.grid(row=2, column=1, padx=5)

tk.Label(input_frame_bi, text="b:", fg="white", bg="#1E1E1E").grid(row=3, column=0, sticky='w')
entry_b_bi = tk.Entry(input_frame_bi, width=10, bg="#2E2E2E", fg="white")
entry_b_bi.grid(row=3, column=1, padx=5)

tk.Label(input_frame_bi, text="Tolerancia:", fg="white", bg="#1E1E1E").grid(row=4, column=0, sticky='w')
entry_tol_bi = tk.Entry(input_frame_bi, width=10, bg="#2E2E2E", fg="white")
entry_tol_bi.grid(row=4, column=1, padx=5)

tk.Label(input_frame_bi, text="Iteraciones:", fg="white", bg="#1E1E1E").grid(row=5, column=0, sticky='w')
entry_iter_bi = tk.Entry(input_frame_bi, width=10, bg="#2E2E2E", fg="white")
entry_iter_bi.grid(row=5, column=1, padx=5)

tk.Button(input_frame_bi, text="Calcular", command=calculate_bisection, bg="#3A3A3A", fg="white").grid(row=6, column=0, columnspan=2, pady=10)

# Frame de tabla de iteraciones
table_frame_bi = tk.Frame(frame_bi, bg="#1E1E1E")
table_frame_bi.pack(fill=tk.X, pady=5)
columns = ("Iteración", "a", "b", "c", "f(c)")
table_bi = ttk.Treeview(table_frame_bi, columns=columns, show="headings", height=5)
for col in columns:
    table_bi.heading(col, text=col)
    table_bi.column(col, width=100)
table_bi.pack(fill=tk.X)

# Frame del gráfico
graph_frame_bi = tk.Frame(frame_bi, bg="#1E1E1E", height=400)
graph_frame_bi.pack(fill=tk.BOTH, expand=True)

# Resultado
result_frame_bi = tk.Frame(frame_bi, bg="#1E1E1E")
result_frame_bi.pack(pady=10)
tk.Label(result_frame_bi, text="Raíz Encontrada", fg="white", bg="#1E1E1E", font=("Arial", 14)).pack()
root_value_bi = tk.Label(result_frame_bi, text="", fg="white", bg="#2E2E2E", font=("Arial", 12), width=20, relief=tk.SUNKEN)
root_value_bi.pack(pady=10, ipadx=10, ipady=10)

# ----- Pestaña Secante -----
frame_sec = tk.Frame(notebook, bg="#1E1E1E")
notebook.add(frame_sec, text="Secante")

input_frame_sec = tk.Frame(frame_sec, bg="#1E1E1E")
input_frame_sec.pack(fill=tk.X, pady=5)

tk.Label(input_frame_sec, text="Altura del gráfico en Y:", fg="white", bg="#1E1E1E").grid(row=0, column=0, sticky='w')
entry_c_sec = tk.Entry(input_frame_sec, width=10, bg="#2E2E2E", fg="white")
entry_c_sec.grid(row=0, column=1, padx=5)

tk.Label(input_frame_sec, text="Función f(x):", fg="white", bg="#1E1E1E").grid(row=1, column=0, sticky='w')
function_entry_sec = tk.Entry(input_frame_sec, width=30, bg="#2E2E2E", fg="white")
function_entry_sec.grid(row=1, column=1, padx=5)

tk.Label(input_frame_sec, text="x0:", fg="white", bg="#1E1E1E").grid(row=2, column=0, sticky='w')
entry_x0_sec = tk.Entry(input_frame_sec, width=10, bg="#2E2E2E", fg="white")
entry_x0_sec.grid(row=2, column=1, padx=5)

tk.Label(input_frame_sec, text="x1:", fg="white", bg="#1E1E1E").grid(row=3, column=0, sticky='w')
entry_x1_sec = tk.Entry(input_frame_sec, width=10, bg="#2E2E2E", fg="white")
entry_x1_sec.grid(row=3, column=1, padx=5)

tk.Label(input_frame_sec, text="Tolerancia:", fg="white", bg="#1E1E1E").grid(row=4, column=0, sticky='w')
entry_tol_sec = tk.Entry(input_frame_sec, width=10, bg="#2E2E2E", fg="white")
entry_tol_sec.grid(row=4, column=1, padx=5)

tk.Label(input_frame_sec, text="Iteraciones:", fg="white", bg="#1E1E1E").grid(row=5, column=0, sticky='w')
entry_iter_sec = tk.Entry(input_frame_sec, width=10, bg="#2E2E2E", fg="white")
entry_iter_sec.grid(row=5, column=1, padx=5)

tk.Button(input_frame_sec, text="Calcular", command=calculate_secant, bg="#3A3A3A", fg="white").grid(row=6, column=0, columnspan=2, pady=10)

table_frame_sec = tk.Frame(frame_sec, bg="#1E1E1E")
table_frame_sec.pack(fill=tk.X, pady=5)
# Se reutilizan las mismas columnas que en Bisección para Secante
table_sec = ttk.Treeview(table_frame_sec, columns=columns, show="headings", height=5)
for col in columns:
    table_sec.heading(col, text=col)
    table_sec.column(col, width=100)
table_sec.pack(fill=tk.X)

graph_frame_sec = tk.Frame(frame_sec, bg="#1E1E1E", height=400)
graph_frame_sec.pack(fill=tk.BOTH, expand=True)

result_frame_sec = tk.Frame(frame_sec, bg="#1E1E1E")
result_frame_sec.pack(pady=10)
tk.Label(result_frame_sec, text="Raíz Encontrada", fg="white", bg="#1E1E1E", font=("Arial", 14)).pack()
root_value_sec = tk.Label(result_frame_sec, text="", fg="white", bg="#2E2E2E", font=("Arial", 12), width=20, relief=tk.SUNKEN)
root_value_sec.pack(pady=10, ipadx=10, ipady=10)

# ----- Pestaña Newton-Raphson -----
frame_newt = tk.Frame(notebook, bg="#1E1E1E")
notebook.add(frame_newt, text="Newton-Raphson")

input_frame_newt = tk.Frame(frame_newt, bg="#1E1E1E")
input_frame_newt.pack(fill=tk.X, pady=5)

tk.Label(input_frame_newt, text="Altura del gráfico en Y:", fg="white", bg="#1E1E1E").grid(row=0, column=0, sticky='w')
entry_c_newt = tk.Entry(input_frame_newt, width=10, bg="#2E2E2E", fg="white")
entry_c_newt.grid(row=0, column=1, padx=5)

tk.Label(input_frame_newt, text="Función f(x):", fg="white", bg="#1E1E1E").grid(row=1, column=0, sticky='w')
function_entry_newt = tk.Entry(input_frame_newt, width=30, bg="#2E2E2E", fg="white")
function_entry_newt.grid(row=1, column=1, padx=5)

tk.Label(input_frame_newt, text="x0:", fg="white", bg="#1E1E1E").grid(row=2, column=0, sticky='w')
entry_x0_newt = tk.Entry(input_frame_newt, width=10, bg="#2E2E2E", fg="white")
entry_x0_newt.grid(row=2, column=1, padx=5)

tk.Label(input_frame_newt, text="Tolerancia:", fg="white", bg="#1E1E1E").grid(row=3, column=0, sticky='w')
entry_tol_newt = tk.Entry(input_frame_newt, width=10, bg="#2E2E2E", fg="white")
entry_tol_newt.grid(row=3, column=1, padx=5)

tk.Label(input_frame_newt, text="Iteraciones:", fg="white", bg="#1E1E1E").grid(row=4, column=0, sticky='w')
entry_iter_newt = tk.Entry(input_frame_newt, width=10, bg="#2E2E2E", fg="white")
entry_iter_newt.grid(row=4, column=1, padx=5)

tk.Button(input_frame_newt, text="Calcular", command=calculate_newton, bg="#3A3A3A", fg="white").grid(row=5, column=0, columnspan=2, pady=10)

# Definición de columnas específicas para Newton-Raphson
columns_newt = ("Iteración", "xₙ", "xₙuevo", "f(xₙ)")
table_frame_newt = tk.Frame(frame_newt, bg="#1E1E1E")
table_frame_newt.pack(fill=tk.X, pady=5)
table_newt = ttk.Treeview(table_frame_newt, columns=columns_newt, show="headings", height=5)
for col in columns_newt:
    table_newt.heading(col, text=col)
    table_newt.column(col, width=120)
table_newt.pack(fill=tk.X)

graph_frame_newt = tk.Frame(frame_newt, bg="#1E1E1E", height=400)
graph_frame_newt.pack(fill=tk.BOTH, expand=True)

result_frame_newt = tk.Frame(frame_newt, bg="#1E1E1E")
result_frame_newt.pack(pady=10)
tk.Label(result_frame_newt, text="Raíz Encontrada", fg="white", bg="#1E1E1E", font=("Arial", 14)).pack()
root_value_newt = tk.Label(result_frame_newt, text="", fg="white", bg="#2E2E2E", font=("Arial", 12), width=20, relief=tk.SUNKEN)
root_value_newt.pack(pady=10, ipadx=10, ipady=10)

plotter = FunctionPlotter()
root.mainloop()
