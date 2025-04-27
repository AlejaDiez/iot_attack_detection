import { Packet } from "@models/packet";
import { ModelsService } from "@services/models.service";
import { LayersModel } from "@tensorflow/tfjs";

/**
 * Modelo de redes neuronal.
 */
export interface Model {
    /** Identificador del modelo */
    id: string;
    /** Nombre del modelo */
    name: string;
}

/**
 * Modelo de Tensorflow.
 */
export interface TensorflowModel extends Model {
    /** Modelo de Tensorflow */
    model: LayersModel;
}

/**
 * Modelos que se pueden usar.
 */
export type Models = Model[];

/**
 * Modelos de Tensorflow.
 */
export type TensorflowModels = TensorflowModel[];

/**
 * Modelos de redes neuronales.
 */
export class CyberShield {
    /** Estado del cyber shield. */
    public enabled: boolean = false;
    /** Modelos de redes neuronales. */
    private _models?: { [model: string]: TensorflowModel };
    /** Modelo que se esta usando. */
    private _model?: TensorflowModel;
    /** Modelo que se esta usando. */
    public get model(): Model | undefined {
        if (this._model) return { id: this._model.id, name: this._model.name };
        else return undefined;
    }
    /** Modelo que se esta usando. */
    public set model(model: string | undefined) {
        if (model && this._models) this._model = this._models[model];
        else this._model = undefined;
    }
    /** Comandos que se pueden realizar. */
    public get availableModels(): Readonly<Models> {
        return this._models
            ? Object.values(this._models).map(({ id, name }) => ({ id, name }))
            : [];
    }

    /**
     * Crea una nueva instancia de CyberShield.
     */
    public constructor() {
        const models = ModelsService.instance;

        this.loadModels(models.models);
        models.models$.subscribe((models) => this.loadModels(models));
    }

    /**
     * Cargar los modelos.
     *
     * @param models Modelos a cargar
     */
    public loadModels(models?: TensorflowModels): void {
        if (models) {
            this._models = {};
            for (const model of models) this._models[model.id] = model;
        } else this._models = undefined;
    }

    /**
     * Se encarga de analizar el paquete con el modelo.
     *
     * @param packet Paquete a analizar
     * @returns Resultado del analisis
     */
    public analyze(packet: Packet): void {}

    /**
     * Convierte el ciber shield a un objeto plano.
     *
     * @returns Objeto plano con los datos del ciber shield.
     */
    public toObject(): any {
        return {
            enabled: this.enabled,
            model: this.model?.id,
        };
    }
}
