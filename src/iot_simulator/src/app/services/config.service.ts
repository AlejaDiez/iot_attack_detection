import { Injectable } from "@angular/core";
import { BehaviorSubject } from "rxjs";

/**
 * Clase que permite gestionar los estados de la aplicación.
 */
class StateManager {
    /** Número máximo de estados a almacenar. */
    private static readonly MAX_STATES: number = 20;
    /** Lista de estados almacenados. */
    private _states: any[];
    /** Índice del estado actual. */
    private _index: number;
    /** Observable para emitir el estado actual y que el NetworkManageService pueda suscribirse. */
    private readonly _stateSubject: BehaviorSubject<any>;
    /** Estado actual. */
    public get state(): any {
        return this._states[this._index];
    }
    /** Mostrar el estado actual como un observable. */
    public get state$() {
        return this._stateSubject.asObservable();
    }
    /** Permite obtener si se puede deshacer un cambio. */
    public get canUndo(): boolean {
        return this._index > 0;
    }
    /** Permite obtener si se puede rehacer un cambio. */
    public get canRedo(): boolean {
        return this._index < this._states.length - 1;
    }

    /**
     * Constructor del gestor de estados.
     */
    public constructor() {
        const state = sessionStorage.getItem("state");

        if (state) this._states = [JSON.parse(state)];
        else this._states = [{}];
        this._index = 0;
        this._stateSubject = new BehaviorSubject<any>(
            this._states[this._index],
        );
    }

    /**
     * Manda un nuevo estado al observable.
     *
     * @param notify Notificar al suscriptor del estado actual.
     */
    private _stateChange(notify: boolean) {
        if (notify) this._stateSubject.next(this.state);
        sessionStorage.setItem("state", JSON.stringify(this.state));
    }

    /**
     * Establece un nuevo estado y lo limita a cierto número de elementos.
     *
     * @param state Nuevo estado a almacenar.
     */
    public setState(state: any, notify: boolean = true) {
        // Eliminar estados futuros
        this._states = this._states.slice(0, this._index + 1);
        // Agregamos el nuevo estado
        this._states.push(state);
        this._index++;
        // Eliminamos el más antiguo en caso de que se exceda el límite
        if (this._states.length > StateManager.MAX_STATES) {
            this._states.shift();
            this._index--;
        }
        // Notificamos el cambio
        this._stateChange(notify);
    }

    public replaceState(state: any, notify: boolean = true) {
        // Reemplazamos el estado actual
        this._states[this._index] = state;
        // Notificamos el cambio
        this._stateChange(notify);
    }

    /**
     * Deshace el último cambio.
     */
    public undo(notify: boolean = true) {
        if (this.canUndo) {
            // Decrementamos el índice
            this._index--;
            // Notificamos el cambio
            this._stateChange(notify);
        }
    }

    /**
     * Rehace el último cambio
     */
    public redo(notify: boolean = true) {
        if (this.canRedo) {
            // Incrementamos el índice
            this._index++;
            // Notificamos el cambio
            this._stateChange(notify);
        }
    }

    /**
     * Reinicia el gestor de estados.
     */
    public reset(notify: boolean = true) {
        // Limpiamos la lista de estados y el índice
        this._states = [{}];
        this._index = 0;
        // Notificamos el cambio
        this._stateChange(notify);
    }
}

@Injectable({ providedIn: "root" })
export class ConfigService {
    /** Gestor de estados de la aplicación. */
    public readonly stateManager: StateManager = new StateManager();

    /**
     * Abre un archivo existente.
     *
     * @param extensions Extensiones de archivo a abrir.
     * @returns Archivo abierto.
     */
    public async openFile(extensions?: string): Promise<string> {
        const input: HTMLInputElement = document.createElement("input");

        input.type = "file";
        input.value = "";
        if (extensions) input.accept = extensions;
        return new Promise<string>((resolve, reject) => {
            input.onchange = async (event: Event) => {
                const file = (event.target as HTMLInputElement).files?.[0];

                // Esperamos un segundo para que se pueda cargar el archivo
                await new Promise((_) => setTimeout(_, 1000));
                // Cargamos el archivo
                if (file) {
                    const reader = new FileReader();

                    reader.onload = () => resolve(reader.result as string);
                    reader.readAsText(file);
                } else reject();
                input.remove();
            };
            // Abrir el diálogo de selección de archivos
            input.click();
        });
    }

    /**
     * Guarda un archivo.
     *
     * @param name Nombre del archivo a guardar.
     * @param value Contenido del archivo a guardar.
     * @param type Tipo de archivo a guardar.
     */
    public async saveFile(
        name: string,
        value: string,
        type?: string,
    ): Promise<void> {
        const fileBlob: Blob = new Blob([value], { type });
        const link: HTMLAnchorElement = document.createElement("a");

        link.href = URL.createObjectURL(fileBlob);
        link.download = name;
        // Esperamos un segundo para que se pueda descargar el archivo
        await new Promise((_) => setTimeout(_, 1000));
        // Descargamos el archivo
        link.click();
        URL.revokeObjectURL(link.href);
        link.remove();
    }
}
