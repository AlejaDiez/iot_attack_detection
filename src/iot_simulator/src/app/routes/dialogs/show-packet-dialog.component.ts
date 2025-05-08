import { CommonModule } from "@angular/common";
import { Component, inject } from "@angular/core";
import { TranslateModule } from "@ngx-translate/core";
import { ConfigService } from "@services/config.service";
import { BrnDialogRef, injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectImports } from "@spartan-ng/brain/select";
import {
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,
} from "@spartan-ng/ui-dialog-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

export interface ShowPacketDialogContext {
    packet: any;
}

@Component({
    selector: "app-show-packet-dialog",
    imports: [
        CommonModule,
        TranslateModule,
        BrnSelectImports,
        HlmDialogHeaderComponent,
        HlmDialogTitleDirective,
        HlmSelectModule,
    ],
    template: `
        <hlm-dialog-header>
            <h3 hlmDialogTitle>{{ "PACKET_DETAILS" | translate }}</h3>
        </hlm-dialog-header>
        <div class="grid gap-4 py-4">
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_SRC_IP" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.srcIP }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_DST_IP" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.dstIP }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_SRC_PORT" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.srcPort }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_DST_PORT" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.dstPort }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_TRANSPORT_PROTOCOL" | translate }}</label
                >
                <span class="col-span-3">{{
                    context.packet.transportProtocol
                }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_APPLICATION_PROTOCOL" | translate }}</label
                >
                <span class="col-span-3">{{
                    context.packet.applicationProtocol
                }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_PAYLOAD" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.payload }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_TOTAL_BYTES" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.totalBytes }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_HEADER_SIZE" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.headerSize }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_PAYLOAD_SIZE" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.payloadSize }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_TIMESTAMP" | translate }}</label
                >
                <span class="col-span-3">{{
                    context.packet.timestamp
                        | date: "medium" : undefined : config.language()
                }}</span>
            </div>
            <div class="grid grid-cols-4 items-center gap-4">
                <label
                    hlmLabel
                    class="items-center text-right text-sm font-medium text-muted-foreground"
                    >{{ "PACKET_TTL" | translate }}</label
                >
                <span class="col-span-3">{{ context.packet.ttl }}</span>
            </div>
            @if (context.packet.type) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_TYPE" | translate }}</label
                    >
                    <span class="col-span-3">{{ context.packet.type }}</span>
                </div>
            }
            @if (context.packet.code) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_CODE" | translate }}</label
                    >
                    <span class="col-span-3">{{ context.packet.code }}</span>
                </div>
            }
            @if (context.packet.identifier) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_IDENTIFIER" | translate }}</label
                    >
                    <span class="col-span-3">{{
                        context.packet.identifier
                    }}</span>
                </div>
            }
            @if (context.packet.sequence) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_SEQUENCE" | translate }}</label
                    >
                    <span class="col-span-3">{{
                        context.packet.sequence
                    }}</span>
                </div>
            }
            @if (context.packet.tcpFlags) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_TCP_FLAGS" | translate }}</label
                    >
                    <span class="col-span-3">{{
                        context.packet.tcpFlags
                    }}</span>
                </div>
            }
            @if (context.packet.ack) {
                <div class="grid grid-cols-4 items-center gap-4">
                    <label
                        hlmLabel
                        class="items-center text-right text-sm font-medium text-muted-foreground"
                        >{{ "PACKET_ACK" | translate }}</label
                    >
                    <span class="col-span-3">{{ context.packet.ack }}</span>
                </div>
            }
        </div>
    `,
    host: { class: "min-w-96 flex flex-col gap-4" },
})
export class ShowPacketDialogComponent {
    public readonly config = inject(ConfigService);
    protected readonly ref: BrnDialogRef<ShowPacketDialogComponent> = inject(
        BrnDialogRef<ShowPacketDialogComponent>,
    );
    protected readonly context =
        injectBrnDialogContext<ShowPacketDialogContext>();

    public ngOnInit(): void {}
}
