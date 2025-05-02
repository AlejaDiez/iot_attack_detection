import { Packet, TransportProtocol } from "@models/packet";
import { ModelsService } from "@services/models.service";
import { StateService } from "@services/state.service";
import { LayersModel, Tensor, tensor } from "@tensorflow/tfjs";

/** Tiempo de espera para el flujo. */
export const FLOW_TIMEOUT = 60;

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
    private _enabled: boolean;
    /** Estado del cyber shield. */
    public get enabled(): boolean {
        return this._enabled;
    }
    /** Estado del cyber shield. */
    @StateService.SetState
    public set enabled(value: boolean) {
        if (this._models && Object.keys(this._models).length > 0)
            this._enabled = value;
        else this._enabled = false;
    }
    /** Modelos de redes neuronales. */
    private _models?: { [model: string]: TensorflowModel };
    /** Modelos de redes neuronales. */
    public get models(): Readonly<Models> {
        return this._models
            ? Object.values(this._models).map(({ id, name }) => ({ id, name }))
            : [];
    }
    /** Modelo que se esta usando. */
    private _model?: TensorflowModel;
    /** Modelo que se esta usando. */
    public get model(): string | undefined {
        return this._model?.id;
    }
    /** Modelo que se esta usando. */
    public set model(model: string | undefined) {
        if (model && this._models) this._model = this._models[model];
        else this._model = undefined;
    }

    /**
     * Crea una instancia de CyberShield.
     */
    public constructor() {
        this._enabled = false;
        this._model = undefined;
        this._models = undefined;

        this.loadModels(ModelsService.instance.models);
        ModelsService.instance.models$.subscribe((models) =>
            this.loadModels(models),
        );
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

    private _flows: any = {};
    /**
     * Se encarga de analizar el paquete con el modelo.
     *
     * @param packet Paquete a analizar
     * @returns Resultado del analisis
     */
    public analyze(packet: Packet): void {
        if (!this.enabled && !this._model) return;

        const dataTransform = (features: any): Tensor => {
            const scaler = {
                mean: [
                    7.586667574904364e-5, -0.0002569646862450844,
                    -0.00034248109469571097, -0.00026144530264210394,
                    0.00010568871645037674, 0.00011832249847384199,
                    -0.00012334874518929518, 0.000588866524575801,
                    -0.00019046289428017218, -0.000383872813645464,
                ],
                std: [
                    0.9999199009295211, 0.999662646579506, 0.9992552509373918,
                    0.9997430875520665, 1.0210691611604972, 0.9915003923723357,
                    0.9572901848819664, 1.1448989638166864, 0.9998521146396753,
                    0.9980514877003535,
                ],
            };
            // Convertir los datos a tensores
            const meanTensor = tensor(scaler.mean);
            const stdTensor = tensor(scaler.std);
            const featuresTensor = tensor(features);

            // Realizar la transformación usando la fórmula (X - mean) / std
            return featuresTensor.sub(meanTensor).div(stdTensor);
        };
        const updateFlow = (packet: Packet) => {
            let {
                srcIP,
                dstIP,
                srcPort,
                dstPort,
                transportProtocol: protocol,
                totalBytes: packetLength,
                payloadSize: outBytes,
                tcpFlags,
            } = packet as any;
            if (
                ![TransportProtocol.TCP, TransportProtocol.UDP].includes(
                    protocol,
                )
            ) {
                srcPort = 0;
                dstPort = 0;
            }
            tcpFlags = tcpFlags ?? 0;
            const flowKey = `${srcIP}_${dstIP}_${srcPort}_${dstPort}_${protocol}`;
            const currentTime = new Date().getTime();
            const flowDuration = currentTime * 1000;

            if (this._flows[flowKey]) {
                const flow = this._flows[flowKey];
                flow.lastSeen = currentTime;
                flow.packetCount += 1;
                flow.totalBytes += packetLength;
                flow.outBytes += outBytes;
                flow.tcpFlags |= tcpFlags;
            } else {
                this._flows[flowKey] = {
                    startTime: currentTime,
                    lastSeen: currentTime,
                    packetCount: 1,
                    totalBytes: packetLength,
                    outBytes: outBytes,
                    tcpFlags: tcpFlags,
                };
            }
        };
        const extractFlowFeatures = () => {
            const closedFlows = [];
            const currentTime = new Date().getTime();

            for (const key in this._flows) {
                const flow = this._flows[key];

                if (currentTime - flow.lastSeen > FLOW_TIMEOUT) {
                    const duration = (flow.lastSeen - flow.startTime) * 1000;
                    const flowKey = key.split("_");
                    const features = [
                        Number(flowKey[2]), // L4_SRC_PORT
                        Number(flowKey[3]), // L4_DST_PORT
                        Number(flowKey[4]), // PROTOCOL
                        0, // L7_PROTO (placeholder)
                        flow.totalBytes, // IN_BYTES
                        flow.outBytes, // OUT_BYTES
                        flow.packetCount, // IN_PKTS
                        flow.packetCount, // OUT_PKTS
                        flow.tcpFlags, // TCP_FLAGS
                        duration, // FLOW_DURATION_MILLISECONDS
                    ];
                    closedFlows.push(features);
                    delete this._flows[key];
                }
            }
            return closedFlows;
        };
        const predictAttack = (model: LayersModel, inputData: Tensor) =>
            (model.predict(inputData) as Tensor).arraySync();

        // Analizar el paquete
        updateFlow(packet);
        const closedFlows = extractFlowFeatures();

        if (closedFlows.length > 0) {
            const inputData = dataTransform(closedFlows);
            const scores = predictAttack(this._model!.model, inputData);

            console.log((scores as number[])[0]);
        }
    }

    /**
     * Convierte el cyber shield a un objeto plano.
     *
     * @returns Objeto plano con los datos del cyber shield.
     */
    public toObject(): any {
        return {
            enabled: this.enabled,
            model: this.model,
        };
    }

    /**
     * Convierte un objeto plano en un cyber shield.
     *
     * @param object Objeto plano a convertir.
     * @returns CyberShield convertido.
     */
    public static fromObject(object: any): CyberShield {
        const cyberShield = new CyberShield();

        // Establecer propiedades
        if (object.enabled) cyberShield._enabled = object.enabled;
        if (object.model && cyberShield._models)
            cyberShield._model = cyberShield._models[object.model];
        return cyberShield;
    }
}
