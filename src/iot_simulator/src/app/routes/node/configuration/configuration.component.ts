import { CommonModule } from "@angular/common";
import { Component, OnInit } from "@angular/core";
import { FormControl, FormGroup, ReactiveFormsModule } from "@angular/forms";
import { ActivatedRoute } from "@angular/router";
import { Node, NodeType } from "@models/node";
import { NetworkManagerService } from "@services/network-manager.service";
import { BrnSelectModule } from "@spartan-ng/brain/select";
import { HlmInputModule } from "@spartan-ng/ui-input-helm";
import { HlmLabelModule } from "@spartan-ng/ui-label-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";
import { HlmSwitchModule } from "@spartan-ng/ui-switch-helm";

@Component({
    imports: [
        CommonModule,
        ReactiveFormsModule,
        BrnSelectModule,
        HlmInputModule,
        HlmLabelModule,
        HlmSelectModule,
        HlmSwitchModule,
    ],
    templateUrl: "configuration.component.html",
})
export class ConfigurationComponent implements OnInit {
    protected readonly types: string[] = [NodeType.IOT, NodeType.COMPUTER];
    protected readonly detection_models: string[] = [
        "tensorflow",
        "federated learning",
    ];
    protected node!: Node;
    protected form: FormGroup = new FormGroup({
        name: new FormControl(""),
        ip: new FormControl({ value: "", disabled: true }),
        type: new FormControl(""),
        detection: new FormControl(true),
        detection_model: new FormControl("tensorflow"),
    });

    public constructor(
        private _route: ActivatedRoute,
        private _networkManager: NetworkManagerService,
    ) {}

    public ngOnInit(): void {
        this._route.parent!.params.subscribe(({ mac }) => {
            this.node = this._networkManager.findByMac(mac);
            this.form.get("name")!.setValue(this.node.name);
            this.form.get("ip")!.setValue(this.node.ip);
            this.form.get("type")!.setValue(this.node.type);
            if (this.node.type === NodeType.ROUTER) {
                this.form.get("type")!.disable();
            }
        });
    }
}
